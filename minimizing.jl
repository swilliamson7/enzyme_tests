using Enzyme, Checkpointing
using ShallowWaters#main

Enzyme.API.maxtypeoffset!(500)
# Enzyme.API.maxtypedepth!(3)
Enzyme.API.runtimeActivity!(true)

using Parameters

function min_not_checkpointed_integration(S)

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
    # @inbounds for i in eachindex(η)
    #     Diag.VolumeFluxes.h[i] = η[i] + S.forcing.H[i]
    # end
    # m, n = size(Diag.VolumeFluxes.h_u)
    # @inbounds for j ∈ 1:n
    #     for i ∈ 1:m
    #         Diag.VolumeFluxes.h_u[i,j] = Float32(0.5)*
    #             (Diag.VolumeFluxes.h[i+1,j] + Diag.VolumeFluxes.h[i,j])
    #     end
    # end
    # m,n = size(Diag.VolumeFluxes.h_v)
    # @inbounds for j ∈ 1:n
    #     for i ∈ 1:m
    #         Diag.VolumeFluxes.h_v[i,j] = Float32(0.5)*
    #             (Diag.VolumeFluxes.h[i,j+1] + Diag.VolumeFluxes.h[i,j])
    #     end
    # end
    # m,n = size(Diag.Vorticity.h_q)
    # @inbounds for j in 1:n, i in 1:m
    #     Diag.Vorticity.h_q[i,j] = Float32(0.25)*(Diag.VolumeFluxes.h[i,j] + 
    #             Diag.VolumeFluxes.h[i+1,j]) + 
    #             Float32(0.25)*(Diag.VolumeFluxes.h[i,j+1] + Diag.VolumeFluxes.h[i+1,j+1])
    # end

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # run integration loop with checkpointing
    return min_not_loop(S)

end

function min_not_loop(S)

    # for S.parameters.i = 1:S.grid.nt

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

        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        for rki = 1:RKo

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                @unpack h,h_u,h_v,U,V = Diag.VolumeFluxes
                @unpack H = S.forcing
                @unpack ep = S.grid

                ShallowWaters.UVfluxes!(u1rhs,v1rhs,η1rhs,Diag,S)
                ShallowWaters.advection_coriolis!(u1rhs,v1rhs,η1rhs,Diag,S)
                ShallowWaters.PVadvection!(Diag,S)

                # adding the terms
                # ShallowWaters.momentum_u!(Diag,S,t)
                ShallowWaters.momentum_v!(Diag,S,t)

                @unpack U,V,dUdx,dVdy = Diag.VolumeFluxes
                @unpack nstep_advcor = S.grid
                @unpack time_scheme,surface_relax,surface_forcing = S.parameters

                # # divergence of mass flux
                ShallowWaters.∂x!(dUdx,U)
                ShallowWaters.∂y!(dVdy,V)

                @unpack dη = Diag.Tendencies
                m,n = size(dη) .- (2,2)     # cut off halo

                @inbounds for j ∈ 1:n
                    for i ∈ 1:m
                        dη[i+1,j+1] = -(Float32(dUdx[i,j+1]) + Float32(dVdy[i+1,j]))
                    end
                end

                if rki < RKo
                    ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                    ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                    ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
                end

                ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
        
        end
        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    # end

    return S.Prog.η[25,25]

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
    nx=50,
    Ndays = 50
    )

    dS = Enzyme.make_zero(S)
    dS.Prog.η[25,25] = 1.0

    @time autodiff(Enzyme.ReverseWithPrimal, min_not_checkpointed_integration, Duplicated(S, dS))

    enzyme_deriv = dS.Prog.u[25,25]

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
    nx=50,
    Ndays = 50
    )

    Jouter = min_not_checkpointed_integration(S_outer)

    diffs2 = []

    for s in steps

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
        nx=50,
        Ndays = 50
        )

        S_inner.Prog.u[25, 25] += s

        Jinner = min_not_checkpointed_integration(S_inner)

        push!(diffs2, (Jinner - Jouter) / s)

    end

    return diffs2, enzyme_deriv

end

diffs2, enzyme = finite_differences()
