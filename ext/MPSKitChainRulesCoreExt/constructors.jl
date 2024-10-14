@non_differentiable MPSKit.updatetol(args...)

function ChainRulesCore.rrule(::Type{MPSKit.PeriodicArray}, data::AbstractArray{T, N}) where {T,N}
    function pullback(Δ)
        return NoTangent(), Δ isa MPSKit.PeriodicArray ? Δ.data : Δ
    end
    return MPSKit.PeriodicArray(data), pullback
end

function ChainRulesCore.rrule(::Type{MPSKit.Multiline}, data::AbstractVector{T}) where {T} 
    function pullback(Δ)
        return NoTangent(), Δ.data
    end
    return MPSKit.Multiline(data), pullback
end