@non_differentiable updatetol(args...)
@non_differentiable gen_init_fps(args...)

function ChainRulesCore.rrule(::Type{PeriodicArray}, data::AbstractArray{T, N}) where {T,N}
    function pullback(Δ)
        return NoTangent(), Δ isa PeriodicArray ? Δ.data : Δ
    end
    return PeriodicArray(data), pullback
end

function ChainRulesCore.rrule(::Type{Multiline}, data::AbstractVector{T}) where {T} 
    function pullback(Δ)
        return NoTangent(), Δ.data
    end
    return Multiline(data), pullback
end

function ChainRulesCore.rrule(::Type{InfiniteMPS}, AL::PeriodicVector{A}, AR::PeriodicVector{A}, CR::PeriodicVector{B}, AC::PeriodicVector{A}) where {A<:GenericMPSTensor, B<:MPSBondTensor}
    function pullback(Δ)
        return NoTangent(), Δ.AL, Δ.AR, Δ.CR, Δ.AC
    end
    return InfiniteMPS(AL, AR, CR, AC), pullback
end

function ChainRulesCore.rrule(::Type{SingleTransferMatrix}, above::A, middle::B, below::C, isflipped::Bool) where {A<:AbstractTensorMap,B,C<:AbstractTensorMap}
    function pullback(Δ)
        @show typeof(Δ)
        Δ = Δ[1]
        return NoTangent(), Δ.above, Δ.middle, Δ.below, NoTangent()
    end
    return SingleTransferMatrix(above, middle, below, isflipped), pullback
end