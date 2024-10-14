using KrylovKit, LinearAlgebra
using Random, Test
using Zygote, ChainRulesCore
using TensorKit
using TensorKit: ℙ
using MPSKit
using MPSKit: ∂∂AC, fixedpoint, updatetol, MPSMultiline, Multiline

include("setup.jl")
using ..TestSetup

@testset "PeriodicArray" begin
    Random.seed!(42)
    A = TensorMap(randn, ComplexF64, ℂ^2 ← ℂ^2)

    function f(β)
        B = β * PeriodicArray([A, A])
        norm(B)
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "Multiline" begin
    Random.seed!(42)
    # A = TensorMap(randn, ComplexF64, ℂ^2 ← ℂ^2)
    A = rand(2,2)

    function f(β)
        B = [β * A, β * A]
        C = Multiline(B)
        norm(C)
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "dominant eigenvalue" begin
    Random.seed!(1234)

    ψ₀ = InfiniteMPS([ℙ^2], [ℙ^2])
    alg = VUMPS(; tol = 1e-8, verbosity = 1)
    alg_eigsolve = updatetol(alg.alg_eigsolve, 1, 1)
    ψ₀ = convert(MPSMultiline, ψ₀)
    H = force_planar(classical_ising(1.0))
    # @show space(H)
    H = Multiline([H])
    ac = RecursiveVec(ψ₀.AC[:, 1])
    envs=environments(ψ₀, H)

    function f(β)
        H = force_planar(classical_ising(β))
        H = Multiline([H])
        H_AC = ∂∂AC(1, ψ₀, H, envs)
        
        _, ac′ = fixedpoint(H_AC, ac, :LM, alg_eigsolve)

        AC = ac′.vecs[1]
        real(dot(AC, AC)) 
    end
    # Zygote.gradient(f, 1.0)
    @show f(1.0)
    # @show Zygote.gradient(f, 1.0)
end

@testset "plansor" begin
    Random.seed!(42)
    # vecspace = ℂ
    vecspace = ℙ
    leftenv = TensorMap(randn, ComplexF64, (vecspace^2 ⊗ (vecspace^2)') ← vecspace^2)
    rightenv = TensorMap(randn, ComplexF64, (vecspace^2 ⊗ vecspace^2) ← vecspace^2)
    opp = TensorMap(randn, ComplexF64, (vecspace^2 ⊗ vecspace^2) ← (vecspace^2 ⊗ vecspace^2))
    x = TensorMap(randn, ComplexF64, (vecspace^2 ⊗ vecspace^2) ← vecspace^2)

    Zygote.gradient(x) do x
        @plansor y[-1 -2; -3] := leftenv[-1 5; 4] * x[4 2; 1] * opp[5 -2; 2 3] *
        rightenv[1 3; -3]
        return real(dot(y, y))
    end
end

@testset "plansor" begin
    Random.seed!(42)
    vecspace = ℙ # ℂ works
    A = TensorMap(randn, ComplexF64, vecspace^2 ← vecspace^2)
    x = TensorMap(randn, ComplexF64, vecspace^2 ← vecspace^2)

    function f(x)
        @plansor y[-1; -2] := A[-1; 1] * x[1; -2]
        return real(dot(y, y))
    end
    @show f(x)
    @show Zygote.gradient(f, x)
end