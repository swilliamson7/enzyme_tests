using Enzyme, Checkpointing
using ShallowWaters#main

Enzyme.API.maxtypeoffset!(3500)
Enzyme.API.runtimeActivity!(true)

using Parameters

function checkpointed_integration(S, scheme)

    # setup
    Diag = S.Diag
    Prog = S.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    @unpack time_scheme,compensated = S.parameters
    @unpack RKaΔt,RKbΔt = S.constants
    @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    ShallowWaters.PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # run integration loop with checkpointing
    loop(S, scheme)

    return S.parameters.J

end

function loop(S,scheme)

    @checkpoint_struct scheme S for S.parameters.i = 1:S.grid.nt

        Diag = S.Diag
        Prog = S.Prog
    
        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies
    
        @unpack um,vm = Diag.SemiLagrange
    
        @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
        @unpack time_scheme,compensated = S.parameters
        @unpack RKaΔt,RKbΔt = S.constants
        @unpack Δt_Δ,Δt_Δs = S.constants
    
        @unpack nt,dtint = S.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid
        t = S.t
        i = S.parameters.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if time_scheme == "RK"   # classic RK4,3 or 2

            if compensated
                fill!(du_sum,zero(Tprog))
                fill!(dv_sum,zero(Tprog))
                fill!(dη_sum,zero(Tprog))
            end

            for rki = 1:RKo
                if rki > 1
                    ShallowWaters.ghost_points!(u1,v1,η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation

                if rki < RKo
                    ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                    ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                    ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
                end

                if compensated      # accumulate tendencies
                    ShallowWaters.axb!(du_sum,RKaΔt[rki],du)
                    ShallowWaters.axb!(dv_sum,RKaΔt[rki],dv)
                    ShallowWaters.axb!(dη_sum,RKaΔt[rki],dη)
                else    # sum RK-substeps on the go
                    ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                    ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                    ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
                end
            end

            if compensated
                # add compensation term to total tendency
                ShallowWaters.axb!(du_sum,-1,du_comp)
                ShallowWaters.axb!(dv_sum,-1,dv_comp)
                ShallowWaters.axb!(dη_sum,-1,dη_comp)

                ShallowWaters.axb!(u0,1,du_sum)   # update prognostic variable with total tendency
                ShallowWaters.axb!(v0,1,dv_sum)
                ShallowWaters.axb!(η0,1,dη_sum)

                ShallowWaters.dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
                ShallowWaters.dambmc!(dv_comp,v0,v,dv_sum)
                ShallowWaters.dambmc!(dη_comp,η0,η,dη_sum)
            end

        elseif time_scheme == "SSPRK2"  # s-stage 2nd order SSPRK

            for rki = 1:RKs
                if rki > 1
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)        # momentum only

                # the update step
                ShallowWaters.axb!(u1,Δt_Δs,du)       # u1 = u1 + Δt/(s-1)*RHS(u1)
                ShallowWaters.axb!(v1,Δt_Δs,dv)

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)
                ShallowWaters.axb!(η1,Δt_Δs,dη)       # η1 = η1 + Δt/(s-1)*RHS(u1)
            end

            a = 1/RKs
            b = (RKs-1)/RKs
            ShallowWaters.cxayb!(u0,a,u,b,u1)
            ShallowWaters.cxayb!(v0,a,v,b,v1)
            ShallowWaters.cxayb!(η0,a,η,b,η1)

        elseif time_scheme == "SSPRK3"  # s-stage 3rd order SSPRK

            @unpack s,kn,mn,kna,knb,Δt_Δnc,Δt_Δn = S.constants.SSPRK3c

            # if compensated
            #     fill!(du_sum,zero(Tprog))
            #     fill!(dv_sum,zero(Tprog))
            #     fill!(dη_sum,zero(Tprog))
            # end

            for rki = 2:s+1       # number of stages (from 2:s+1 to match Ketcheson et al 2014)
                if rki > 2
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn    # special case combining more previous stages
                    ShallowWaters.dxaybzc!(u1,kna,u1,knb,u0,Δt_Δnc,du)
                    ShallowWaters.dxaybzc!(v1,kna,v1,knb,v0,Δt_Δnc,dv)
                else                                # normal update case
                    ShallowWaters.axb!(u1,Δt_Δn,du)
                    ShallowWaters.axb!(v1,Δt_Δn,dv)

                    # if compensated
                    #     axb!(du_sum,Δt_Δn,du)
                    #     axb!(dv_sum,Δt_Δn,dv)
                    # end
                end

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn
                    ShallowWaters.dxaybzc!(η1,kna,η1,knb,η0,Δt_Δnc,dη)
                else
                    ShallowWaters.axb!(η1,Δt_Δn,dη)
                    # if compensated
                    #     axb!(dη_sum,Δt_Δn,dη)
                    # end
                end

                # special stage that is needed later for the kn-th stage, store in u0,v0,η0 therefore
                # or for the last step, as u0,v0,η0 is used as the last step's result of any RK scheme.
                if rki == mn || rki == s+1
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    ShallowWaters.ghost_points_η!(η1,S)
                    copyto!(η0,η1)
                end
            end

        elseif time_scheme == "4SSPRK3"   # 4-stage SSPRK3

            for rki = 1:4
                if rki > 1
                    ShallowWaters.ghost_points!(u1,v1,η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                ShallowWaters.caxb!(u0,u1,Δt_Δ,du)        # store Euler update into u0,v0
                ShallowWaters.caxb!(v0,v1,Δt_Δ,dv)
                ShallowWaters.cxab!(u1,1/2,u1,u0)         # average u0,u1 and store in u1
                ShallowWaters.cxab!(v1,1/2,v1,v0)         # same

                # semi-implicit for continuity equation, use u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                ShallowWaters.caxb!(η0,η1,Δt_Δ,dη)    # store Euler update into η0
                ShallowWaters.cxab!(η1,1/2,η1,η0)         # average η0,η1 and store in η1

                if rki == 3
                    ShallowWaters.cxayb!(u1,2/3,u,1/3,u1)
                    ShallowWaters.cxayb!(v1,2/3,v,1/3,v1)
                    ShallowWaters.cxayb!(η1,2/3,η,1/3,η1)
                elseif rki == 4
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    copyto!(η0,η1)
                end
            end
        end

        ShallowWaters.ghost_points!(u0,v0,η0,S)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S)
            ShallowWaters.ghost_points_uv!(u0,v0,S)
        end

        t += dtint

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)  # copy back as add_drag_diff_tendencies changed u0,v0
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S)

        #### cost function evaluation

        # if S.parameters.i in S.parameters.data_steps

        #     temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
        #     energy_lr = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)

        #     # spacially averaged energy objective function
        #     S.parameters.J += (energy_lr - S.parameters.data[S.parameters.j])^2

        #     S.parameters.j += 1

        # end

        
        #############################################################

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    # regularization term
    # S.parameters.J += 0.01 * abs(S.parameters.γ₀)

    temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,S.Prog.v,S.Prog.η,S.Prog.sst,S)...)

    # S.parameters.J = temp.v[24,24]

    # S.parameters.J = sum(temp.v.^2)
    # S.parameters.J = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)

    S.parameters.J = temp.η[25,25]

    return nothing

end

function not_checkpointed_integration(S, scheme)

    # setup
    Diag = S.Diag
    Prog = S.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    @unpack time_scheme,compensated = S.parameters
    @unpack RKaΔt,RKbΔt = S.constants
    @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    ShallowWaters.PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # run integration loop with checkpointing
    not_loop(S, scheme)

    return S.parameters.J

end

function not_loop(S,scheme)

    for S.parameters.i = 1:S.grid.nt

        Diag = S.Diag
        Prog = S.Prog
    
        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies
    
        @unpack um,vm = Diag.SemiLagrange
    
        @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
        @unpack time_scheme,compensated = S.parameters
        @unpack RKaΔt,RKbΔt = S.constants
        @unpack Δt_Δ,Δt_Δs = S.constants
    
        @unpack nt,dtint = S.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid
        t = S.t
        i = S.parameters.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if time_scheme == "RK"   # classic RK4,3 or 2

            if compensated
                fill!(du_sum,zero(Tprog))
                fill!(dv_sum,zero(Tprog))
                fill!(dη_sum,zero(Tprog))
            end

            for rki = 1:RKo
                if rki > 1
                    ShallowWaters.ghost_points!(u1,v1,η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation

                if rki < RKo
                    ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                    ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                    ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
                end

                if compensated      # accumulate tendencies
                    ShallowWaters.axb!(du_sum,RKaΔt[rki],du)
                    ShallowWaters.axb!(dv_sum,RKaΔt[rki],dv)
                    ShallowWaters.axb!(dη_sum,RKaΔt[rki],dη)
                else    # sum RK-substeps on the go
                    ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                    ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                    ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
                end
            end

            if compensated
                # add compensation term to total tendency
                ShallowWaters.axb!(du_sum,-1,du_comp)
                ShallowWaters.axb!(dv_sum,-1,dv_comp)
                ShallowWaters.axb!(dη_sum,-1,dη_comp)

                ShallowWaters.axb!(u0,1,du_sum)   # update prognostic variable with total tendency
                ShallowWaters.axb!(v0,1,dv_sum)
                ShallowWaters.axb!(η0,1,dη_sum)

                ShallowWaters.dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
                ShallowWaters.dambmc!(dv_comp,v0,v,dv_sum)
                ShallowWaters.dambmc!(dη_comp,η0,η,dη_sum)
            end

        elseif time_scheme == "SSPRK2"  # s-stage 2nd order SSPRK

            for rki = 1:RKs
                if rki > 1
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)        # momentum only

                # the update step
                ShallowWaters.axb!(u1,Δt_Δs,du)       # u1 = u1 + Δt/(s-1)*RHS(u1)
                ShallowWaters.axb!(v1,Δt_Δs,dv)

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)
                ShallowWaters.axb!(η1,Δt_Δs,dη)       # η1 = η1 + Δt/(s-1)*RHS(u1)
            end

            a = 1/RKs
            b = (RKs-1)/RKs
            ShallowWaters.cxayb!(u0,a,u,b,u1)
            ShallowWaters.cxayb!(v0,a,v,b,v1)
            ShallowWaters.cxayb!(η0,a,η,b,η1)

        elseif time_scheme == "SSPRK3"  # s-stage 3rd order SSPRK

            @unpack s,kn,mn,kna,knb,Δt_Δnc,Δt_Δn = S.constants.SSPRK3c

            # if compensated
            #     fill!(du_sum,zero(Tprog))
            #     fill!(dv_sum,zero(Tprog))
            #     fill!(dη_sum,zero(Tprog))
            # end

            for rki = 2:s+1       # number of stages (from 2:s+1 to match Ketcheson et al 2014)
                if rki > 2
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn    # special case combining more previous stages
                    ShallowWaters.dxaybzc!(u1,kna,u1,knb,u0,Δt_Δnc,du)
                    ShallowWaters.dxaybzc!(v1,kna,v1,knb,v0,Δt_Δnc,dv)
                else                                # normal update case
                    ShallowWaters.axb!(u1,Δt_Δn,du)
                    ShallowWaters.axb!(v1,Δt_Δn,dv)

                    # if compensated
                    #     axb!(du_sum,Δt_Δn,du)
                    #     axb!(dv_sum,Δt_Δn,dv)
                    # end
                end

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn
                    ShallowWaters.dxaybzc!(η1,kna,η1,knb,η0,Δt_Δnc,dη)
                else
                    ShallowWaters.axb!(η1,Δt_Δn,dη)
                    # if compensated
                    #     axb!(dη_sum,Δt_Δn,dη)
                    # end
                end

                # special stage that is needed later for the kn-th stage, store in u0,v0,η0 therefore
                # or for the last step, as u0,v0,η0 is used as the last step's result of any RK scheme.
                if rki == mn || rki == s+1
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    ShallowWaters.ghost_points_η!(η1,S)
                    copyto!(η0,η1)
                end
            end

        elseif time_scheme == "4SSPRK3"   # 4-stage SSPRK3

            for rki = 1:4
                if rki > 1
                    ShallowWaters.ghost_points!(u1,v1,η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                ShallowWaters.caxb!(u0,u1,Δt_Δ,du)        # store Euler update into u0,v0
                ShallowWaters.caxb!(v0,v1,Δt_Δ,dv)
                ShallowWaters.cxab!(u1,1/2,u1,u0)         # average u0,u1 and store in u1
                ShallowWaters.cxab!(v1,1/2,v1,v0)         # same

                # semi-implicit for continuity equation, use u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                ShallowWaters.caxb!(η0,η1,Δt_Δ,dη)    # store Euler update into η0
                ShallowWaters.cxab!(η1,1/2,η1,η0)         # average η0,η1 and store in η1

                if rki == 3
                    ShallowWaters.cxayb!(u1,2/3,u,1/3,u1)
                    ShallowWaters.cxayb!(v1,2/3,v,1/3,v1)
                    ShallowWaters.cxayb!(η1,2/3,η,1/3,η1)
                elseif rki == 4
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    copyto!(η0,η1)
                end
            end
        end

        ShallowWaters.ghost_points!(u0,v0,η0,S)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S)
            ShallowWaters.ghost_points_uv!(u0,v0,S)
        end

        t += dtint

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)  # copy back as add_drag_diff_tendencies changed u0,v0
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S)

        # # feedback and output
        # feedback.i = i
        # feedback!(Prog,feedback,S)
        # ShallowWaters.output_nc!(S.parameters.i,netCDFfiles,Prog,Diag,S)       # uses u0,v0,η0

        # if feedback.nans_detected
        #     break
        # end

        #### cost function evaluation

        # if S.parameters.i in S.parameters.data_steps

        #     temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
        #     energy_lr = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)

        #     # spacially averaged energy objective function
        #     S.parameters.J += (energy_lr - S.parameters.data[S.parameters.j])^2

        #     S.parameters.j += 1

        # end

        
        #############################################################

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    # regularization term
    # S.parameters.J += 0.01 * abs(S.parameters.γ₀)

    temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,S.Prog.v,S.Prog.η,S.Prog.sst,S)...)

    # S.parameters.J = temp.v[24,24]

    # S.parameters.J = sum(temp.v.^2)
    # S.parameters.J = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)

    S.parameters.J = temp.η[25,25]

    return nothing

end

function compute_derivative()

    # first computing the derivative with Checkpointing

    S = model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays = 1
    )

    dS = Enzyme.make_zero(S)
    dS.Prog.η[25,25] = 1.0

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt, snaps; verbose=1, gc=true, write_checkpoints=false)

    @time autodiff(Enzyme.ReverseWithPrimal, checkpointed_integration, Duplicated(S, dS), Const(revolve))

    # now computing the derivative without checkpointing

    S2 = model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays = 1
    )

    dS2 = Enzyme.make_zero(S2)
    dS2.Prog.η[25,25] = 1.0

    snaps2 = Int(floor(sqrt(S2.grid.nt)))
    revolve2 = Revolve{ShallowWaters.ModelSetup}(S2.grid.nt, snaps2; verbose=1, gc=true, write_checkpoints=false)

    @time autodiff(Enzyme.ReverseWithPrimal, not_checkpointed_integration, Duplicated(S2, dS2), Const(revolve2))

    @show S.parameters.J, S2.parameters.J

    @assert abs(S.parameters.J - S2.parameters.J) < 1e-8

    @assert abs(dS.Prog.u[25, 25] - dS2.Prog.u[25,25]) < 1e-8

    return dS, dS2, S, S2

end

function finite_differences()

    S = model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays = 1
    )

    dS = Enzyme.make_zero(S)
    dS.Prog.η[25,25] = 1.0

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt, snaps; verbose=1, gc=true, write_checkpoints=false)

    @time autodiff(Enzyme.ReverseWithPrimal, checkpointed_integration, Duplicated(S, dS), Const(revolve))

    temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,S.Prog.v,S.Prog.η,S.Prog.sst,S)...)
    enzyme_deriv = temp.u[25,25]

    steps = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]

    S_outer = model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays = 1
    )

    Pouter = ShallowWaters.time_integration(S_outer)

    diffs2 = []

    for s in  steps

        S_inner = model_setup(output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays = 1
        )

        S_inner.Prog.u[25, 25] += s

        Pinner = ShallowWaters.time_integration(S_inner)

        push!(diffs2, (Pinner.η[25, 25] - Pouter.η[25, 25]) / s)

    end

    return diffs, enzyme_calculated_derivative

end

# dS, dS2 = compute_derivative()