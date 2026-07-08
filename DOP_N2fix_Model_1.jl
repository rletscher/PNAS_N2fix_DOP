# Linear P cycling model with N2 fixation from Letscher et al at PNAS (in revision)
# Model 1: Iron limitation is function of dissolved [Fe]

using CSV
using DataFrames
using MAT
using F1Method
using AIBECS
import AIBECS: @units, units
import AIBECS: @limits, limits
import AIBECS: @initial_value, initial_value
import AIBECS: @flattenable, flattenable
import AIBECS: @prior, prior
using Unitful: m, d, s, yr, Myr, mol, mmol, μmol, μM, NoUnits
using Distributions
using WorldOceanAtlasTools


# OCIM2 circulation TTM
grd, T_Circ = OCIM2.load()
T_DIP(p) = T_Circ


# sinking particle operator
T_POP(p) = transportoperator(grd, z -> w(z,p))
function w(z,p)
    @unpack w0, w1 = p
    return @. w0 + w1 * z
end

z = depthvec(grd)
z_top = topdepthvec(grd) # define top layer


# load NPP data and vectorize it
file = matopen("NPP.mat")
NPP = read(file, "NPP") # mmol C m-3 yr-1 
NPP_v = vectorize(NPP["NPP"],grd) * 73     # mmol C m-2 yr-1; dz of Ez = 73m
NPP_v = ustrip.(upreferred.(NPP_v * u"mmol/m^2/yr"))

# load WOA PO4 data and vectorize it
file2 = matopen("WOAPO4.mat")
WOAPO4 = read(file2)
WOAPO4_v = vectorize(WOAPO4["WOAPO4"],grd)
WOAPO4_v = ustrip.(upreferred.(WOAPO4_v * u"mmol/m^3"))
PO4_obs = read(file2, "WOAPO4")     # size 3D

# load WOA NO3 data and vectorize it 
file3 = matopen("WOANO3.mat")
WOANO3 = read(file3)
WOANO3_v = vectorize(WOANO3["WOANO3"],grd)
WOANO3_v = ustrip.(upreferred.(WOANO3_v * u"mmol/m^3"))
NO3_obs = read(file3, "WOANO3")     # size 3D

# load SLDOP data product and vectorize it
file4 = matopen("DOP.mat")
DOPsl = read(file4, "DOP") # Letscher et al 2022 OCIM2 DOPsl-cycling output
DOPsl_v = vectorize(DOPsl,grd) 
DOPsl_v = filter(!isnan, DOPsl_v) # remove NaN in DOPsl_v 
DOPsl_v[(DOPsl_v) .< 0] .= 0 # zero out negative values
DOPsl_v = ustrip.(upreferred.(DOPsl_v * u"mmol/m^3"))

# load temperature
file5 = matopen("OCIM2_Temp.mat")
Temp = read(file5, "Temp") # OCIM2 optimized temperature in degC
Temp_v = vectorize(Temp,grd)
Temp_v = filter(!isnan, Temp_v) # remove NaN in Temp_v

# build 3D spatially varying P:C and N:P ratios for phytoplankton biomass
p2c = 0.006 .+ (0.0069 .* PO4_obs) # P:C ratio from Galbraith + Martiny 2015
n2c = 0.125 .+ ((0.030 .* NO3_obs) ./ (0.32 .+ NO3_obs)) # N:C ratio from Galbraith + Martiny 2015
n2p = n2c ./ p2c # spatially variable N:P based on WOA NO3 + PO4 distributions
n2p_v = vectorize(n2p,grd)
n2p_v = filter(!isnan, n2p_v) # remove NaN in n2p_v
p2c_v = vectorize(p2c,grd)
p2c_v = filter(!isnan, p2c_v) # remove NaN in p2c_v

NPP0 = mean(filter(!isnan, p2c_v .* NPP_v))    # reference NPP in P units
DIP0 = mean(filter(!isnan, WOAPO4_v))          # reference DIP

# gamma term: DIP uptake rate parameterized from NPP
function gamma_DIP(x, p)
    @unpack α, β = p
     @. α*((p2c_v * NPP_v / NPP0) / (WOAPO4_v / DIP0))^β * x * (z_top≤73) 
end

# DOP production from NPP
function sigma_DOP(x, p)
    @unpack α, β, σ = p
     @. σ * α * (p2c_v * NPP_v / NPP0 / (WOAPO4_v / DIP0))^β * x * (z_top ≤ 73)
end

# POP production from NPP
function sigma_POP(x, p)
    @unpack α, β, σ = p
     @. (1 - σ) * α * (p2c_v * NPP_v / NPP0 / (WOAPO4_v / DIP0))^β * x * (z_top ≤ 73)
end

# SLDOP autotrophic uptake
Vm = 20.0       # [yr] from Letscher et al 2022
Vm = ustrip.(upreferred.(Vm * u"yr"))
function mu_SLDOP(x, p)
     @unpack Ks = p
      @. ((Ks+WOAPO4_v)/ (WOAPO4_v + 0.000001))/Vm*x*(z_top≤73)
end

#fit a logistic growth curve for the temperature limiter
function t(p)
    @unpack t0 = p
    return 1 ./ (1 .+ exp.( .- (Temp_v .- t0))) .* (Temp_v .< 31.4)
end

# build iron limiter term
file6 = matopen("dFe_onOCIM2.mat")
Fe = read(file6, "dFe") # dissolved Fe output from Pasqiuer + Holzer 2017
Fe_v = vectorize(Fe,grd)
Fe_v = filter(!isnan, Fe_v) # remove NaN in Fe_v
Fe_v[(Fe_v) .< 0] .= 0 # zero out negative values
Fe_v = Fe_v .* 1000.0 # convert µM to nM
Fe_v_norm = (Fe_v .- minimum(Fe_v)) ./ (maximum(Fe_v) .- minimum(Fe_v))   # 0-1 normalization

# SLDOP diazotrophic uptake
function diaz_SLDOP(y, p)
     @unpack Ko = p
     t_val = t(p)
      @. ((Ko+WOAPO4_v) / (WOAPO4_v + 0.000001))/ Vm * y * Fe_v_norm * t_val * (z_top≤73)
end

# DIP diazotrophic uptake
function diaz_DIP(x, p)
    @unpack αd, β = p
    t_val = t(p)
     @. αd * (p2c_v * NPP_v / NPP0 / (WOAPO4_v / DIP0))^β * x * Fe_v_norm * t_val * (z_top ≤ 73)
end

# SLDOP remin
function R_SLDOP(x, p)
     @unpack τ_SLDOP = p
      @. x / τ_SLDOP
end

# POP remin
function R_POP(x, p)
     @unpack τ_POP = p
      @. x / τ_POP
end

# DIP geologic restoring
DIPgeo = 2.12       # mean ocean PO4
DIPgeo = ustrip.(upreferred.(DIPgeo * u"mmol/m^3"))
τ_g = 1.0           # million year restoring timescale
τ_g = ustrip.(upreferred.(τ_g * u"Myr"))

function R_geo_DIP(x)
     @. (DIPgeo - x) / τ_g
end

# POP geologic restoring
function R_geo_POP(x)
     @. x / τ_g
end

# compute DOP-supported N2-fixation
function Jfixo(y, p)
    diaz_SLDOP_val = diaz_SLDOP(y, p)
    return diaz_SLDOP_val
end

# compute DIP-supported N2-fixation
function Jfixi(x, p)
    diaz_DIP_val = diaz_DIP(x, p)
    return diaz_DIP_val
end

# compute total N2-fixation
function Jfix(x, y, p)
    diaz_DIP_val   = diaz_DIP(x, p)
    diaz_SLDOP_val = diaz_SLDOP(y, p)
    return diaz_SLDOP_val .+ diaz_DIP_val
end 

# combine DIP source sink terms
function G_DIP(DIP, DOP, POP, p)
    R_SLDOP(DOP,p) - gamma_DIP(DIP,p) - diaz_DIP(DIP,p) + R_POP(POP,p) + R_geo_DIP(DIP)
end

# combine DOP source sink terms
function G_DOP(DIP, DOP, POP, p)
    sigma_DOP(DIP,p) - R_SLDOP(DOP,p) - mu_SLDOP(DOP,p) - diaz_SLDOP(DOP,p) + 0*POP
end

# combine POP source sink terms
function G_POP(DIP, DOP, POP, p)
    sigma_POP(DIP, p) -  R_POP(POP,p) + mu_SLDOP(DOP,p) + diaz_DIP(DIP,p) + diaz_SLDOP(DOP,p) - R_geo_POP(POP)
end

∞ = Inf

@initial_value @units @flattenable @limits struct N2fixmodelParameters{U} <: AbstractParameters{U}
    τ_SLDOP::U  | 4.4           | yr             | false | (0.5, 25)
    τ_POP::U    | 0.0822        | yr             | false | (0, 0.5)
    α::U        | 0.6           | 1 / yr         | true  | (0.05, 2)
    β::U        | 0.5           | NoUnits        | true  | (0.4, 1)
    σ::U        | 0.21          | NoUnits        | false | (0, 1)
    Ks::U       | 0.2           | mmol/m^3       | false | (0, 2.7)
    Ko::U       | 0.8           | mmol/m^3       | true  | (0.6, 1)
    αd::U       | 0.2           | 1 / yr         | true  | (0.05, 2)
    t0::U       | 15.0          | NoUnits    	| true  | (10, 25)
    w0::U       | 2.0           | m/d            | true  | (0, 50)
    w1::U       | 0.05          | m/d/m          | true  | (0, 0.2)
end

function prior(::Type{T}, s::Symbol) where {T<:AbstractParameters}
    if flattenable(T, s)
        lb, ub = limits(T, s)
        if (lb, ub) == (0, ∞)
            μ = log(initial_value(T, s))
            LogNormal(μ, 1.0)
        elseif (lb, ub) == (-∞, ∞)
            μ = initial_value(T, s)
            σ = 10.0 # Assumes that a sensible unit is chosen (i.e., that within 10.0 * U)
            Distributions.Normal(μ, σ)
        else # LogitNormal with median as initial value and bounds
            m = initial_value(T, s)
            f = (m - lb) / (ub - lb)
            LocationScale(lb, ub - lb, LogitNormal(log(f / (1 - f)), 1.0))
        end
    else
        nothing
    end
end

prior(::T, s::Symbol) where {T<:AbstractParameters} = prior(T, s)
prior(::Type{T}) where {T<:AbstractParameters} = Tuple(prior(T, s) for s in AIBECS.symbols(T))
prior(::T) where {T<:AbstractParameters} = prior(T)

p = N2fixmodelParameters()

nb = sum(iswet(grd))
F = AIBECSFunction((T_DIP, T_DIP, T_POP), (G_DIP, G_DOP, G_POP), nb, N2fixmodelParameters)

x = ustrip(upreferred(1.0mmol/m^3)) * ones(3nb) # initial guess
prob = SteadyStateProblem(F, x, p)



# load N2 fixation observations
const obs_N2 = let
    obs_N2 = DataFrame(CSV.File("N2fixrates_Shao2023_volumetric_nozeros.csv")) #units are mmolN/m3/yr
    obs_N2 = filter(:Jfix => >(0), obs_N2)
    obs_N2.Jfix = obs_N2.Jfix ./ 20.0 # convert to P units with Diaz N:P = 20:1
    obs_N2.Jfix = filter(!isnan, obs_N2.Jfix) # remove NaN in N2.Jfix
    obs_N2.value = ustrip.(upreferred.(obs_N2.Jfix * u"mmol/m^3/yr"))
    obs_N2
end

const obs_PO4 = let
    obs_PO4 = DataFrame(CSV.File("OCIMcoordinate.csv"))
    obs_PO4.PO4 = WOAPO4_v
    obs_PO4.PO4 = filter(!isnan, obs_PO4.PO4) # remove NaN in obs_PO4
    obs_PO4.value = obs_PO4.PO4
    obs_PO4
end

const obs_DOPsl = let
    obs_DOPsl = DataFrame(CSV.File("OCIMcoordinate.csv"))
    obs_DOPsl.DOPsl = DOPsl_v
    obs_DOPsl.DOPsl[obs_DOPsl.DOPsl .<= 0] .= 0.00001  # Assign a small value to zero or negative values
    obs_DOPsl.value = DOPsl_v
    obs_DOPsl
end

const obs = (obs_PO4, obs_DOPsl, obs_N2)

modify(DIP, DOP, POP) = (DIP, DOP, Jfix(DIP, DOP,p))


ωs = (1.0, 1.0, 10.0)     # weight for the observations. DIP, DOP, N2fix
ωp = 1e-4                 # weight for the parameter priors
f, ∇ₓf = f_and_∇ₓf(ωs, ωp, grd, modify, obs, N2fixmodelParameters)

using F1Method
using Distributions

λ = p2λ(p)

τ = ustrip(u"s", 1e3u"Myr")
mem = F1Method.initialize_mem(F, ∇ₓf, x, λ, CTKAlg(), τstop=τ)

function objective(λ)
    p = λ2p(N2fixmodelParameters, λ); @show p
    F1Method.objective(f, F, mem, λ, CTKAlg(), τstop=τ)
end

gradient(λ) = F1Method.gradient(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)
hessian(λ)  = F1Method.hessian(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)

using Optim

opt = Optim.Options(store_trace=false, show_trace=true, extended_trace=false, g_tol=1e-5)

res = optimize(objective, gradient, hessian, λ, NewtonTrustRegion(), opt; inplace=false)

# solve the steady state equation with new p
p_optimized = λ2p(N2fixmodelParameters, res.minimizer)
prob_optimized = SteadyStateProblem(F, x, p_optimized)
s_optimized = solve(prob_optimized, CTKAlg(), τstop=ustrip(s, 1e3Myr)).u

DIP, DOP, POP = state_to_tracers(s_optimized, grd)

hess = hessian(res.minimizer)

# save output
using JLD2
tp_opt = AIBECS.table(p_optimized)

jldsave("output_Model1_20.jld2"; DIP=DIP, DOP=DOP, POP=POP, Jfix=Jfix, tp_opt=tp_opt, hess=hess)

matwrite("pooloutput_Model1_20.mat", Dict(
     "lambda" => res.minimizer .+ 0, # optimized parameters
     "Jfix" => Jfix(DIP, DOP, p_optimized),
     "DIP" => DIP .+ 0,
     "DOP" => DOP .+ 0,
     "POP" => POP .+ 0,
     "Jfixi" => Jfixi(DIP, p_optimized),
     "Jfixo" => Jfixo(DOP, p_optimized),
     "diaz_DIP" => diaz_DIP(DIP, p_optimized),
     "diaz_SLDOP" => diaz_SLDOP(DOP, p_optimized),
     "hess" => hess
 ))

println("Done!")
