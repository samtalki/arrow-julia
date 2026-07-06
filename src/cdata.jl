# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mutable struct ArrowSchema
    format::Cstring
    name::Cstring
    metadata::Cstring
    flags::Int64
    n_children::Int64
    children::Ptr{Ptr{ArrowSchema}}
    dictionary::Ptr{ArrowSchema}
    release::Ptr{Cvoid}
    private_data::Ptr{Cvoid}
end

mutable struct ArrowArray
    length::Int64
    null_count::Int64
    offset::Int64
    n_buffers::Int64
    n_children::Int64
    buffers::Ptr{Ptr{Cvoid}}
    children::Ptr{Ptr{ArrowArray}}
    dictionary::Ptr{ArrowArray}
    release::Ptr{Cvoid}
    private_data::Ptr{Cvoid}
end

const _CDATA_PTR_SIZE = sizeof(Ptr{Cvoid})
@assert sizeof(ArrowSchema) == 7 * _CDATA_PTR_SIZE + 2 * sizeof(Int64)
@assert sizeof(ArrowArray) == 5 * _CDATA_PTR_SIZE + 5 * sizeof(Int64)

const ARROW_FLAG_DICTIONARY_ORDERED = Int64(1)
const ARROW_FLAG_NULLABLE = Int64(2)
const ARROW_FLAG_MAP_KEYS_SORTED = Int64(4)

const _CDATA_MAX_CHILDREN = 10_000
const _CDATA_MAX_DEPTH = 128
const _CDATA_MAX_BUFFERS = 64
const _CDATA_MAX_FORMAT_BYTES = 4096
const _CDATA_MAX_NAME_BYTES = 1 << 16
const _CDATA_MAX_METADATA_PAIRS = 4096
const _CDATA_MAX_METADATA_BYTES = 1 << 20
const _CDATA_MAX_METADATA_FIELD_BYTES = 1 << 20
const _CDATA_MAX_FIXED_SIZE = 4096

abstract type CDataFormat end

struct CDataNullFormat <: CDataFormat end
struct CDataBoolFormat <: CDataFormat end
struct CDataPrimitiveFormat <: CDataFormat
    storage::Type
end
struct CDataBinaryFormat{O} <: CDataFormat
    juliatype::Type
end
struct CDataFixedSizeBinaryFormat <: CDataFormat
    bytewidth::Int
end
struct CDataListFormat{O} <: CDataFormat end
struct CDataFixedSizeListFormat <: CDataFormat
    listsize::Int
end
struct CDataStructFormat <: CDataFormat end

abstract type CDataVector{T} <: ArrowVector{T} end

mutable struct CDataOwner
    schema_ptr::Ptr{ArrowSchema}
    array_ptr::Ptr{ArrowArray}
    released::Bool
    lock::ReentrantLock
end

function CDataOwner(schema_ptr::Ptr{ArrowSchema}, array_ptr::Ptr{ArrowArray})
    owner = CDataOwner(schema_ptr, array_ptr, false, ReentrantLock())
    finalizer(release_c_data, owner)
    return owner
end

struct CDataValidity
    bytes::Vector{UInt8}
    bitoffset::Int
    len::Int
    null_count::Int
end

struct CDataNull{T} <: CDataVector{T}
    owner::CDataOwner
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataPrimitive{T,S,A<:AbstractVector{S}} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    data::A
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataBool{T} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    data::Vector{UInt8}
    bitoffset::Int
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataBinary{T,O,A<:AbstractVector{O}} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    offsets::A
    data::Vector{UInt8}
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataFixedSizeBinary{T} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    data::Vector{UInt8}
    bytewidth::Int
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataList{T,O,A<:AbstractVector{O},C<:AbstractVector} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    offsets::A
    data::C
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataFixedSizeList{T,C<:AbstractVector} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    data::C
    listsize::Int
    offset::Int
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataStruct{T,S,fnames} <: CDataVector{T}
    owner::CDataOwner
    validity::CDataValidity
    data::S
    offset::Int
    len::Int
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
end

struct CDataSlice{T,V<:AbstractVector{T}} <: CDataVector{T}
    parent::V
    first::Int
    len::Int
end

struct CDataTable <: Tables.AbstractColumns
    names::Vector{Symbol}
    types::Vector{Type}
    columns::Vector{AbstractVector}
    lookup::Dict{Symbol,AbstractVector}
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
    owner::CDataOwner
    rowcount::Int
end

struct CDataNode
    schema_ptr::Ptr{ArrowSchema}
    array_ptr::Ptr{ArrowArray}
    schema::ArrowSchema
    array::ArrowArray
    format::CDataFormat
    name::Union{Nothing,String}
    metadata::Union{Nothing,Base.ImmutableDict{String,String}}
    buffers::Vector{Ptr{Cvoid}}
    children::Vector{CDataNode}
    len::Int
    offset::Int
    null_count::Int
end

Base.IndexStyle(::Type{<:CDataVector}) = Base.IndexLinear()

Base.size(x::CDataNull) = (x.len,)
Base.size(x::CDataPrimitive) = size(x.data)
Base.size(x::CDataBool) = (x.len,)
Base.size(x::CDataBinary) = (x.len,)
Base.size(x::CDataFixedSizeBinary) = (x.len,)
Base.size(x::CDataList) = (x.len,)
Base.size(x::CDataFixedSizeList) = (x.len,)
Base.size(x::CDataStruct) = (x.len,)
Base.size(x::CDataSlice) = (x.len,)
Base.length(t::CDataTable) = length(getfield(t, :columns))

_owner(x::CDataVector) = getfield(x, :owner)
_owner(x::CDataSlice) = _owner(x.parent)

function _check_live(owner::CDataOwner)
    lock(owner.lock)
    try
        owner.released && throw(ArgumentError("Arrow C Data object has been released"))
        return
    finally
        unlock(owner.lock)
    end
end

_check_live(x::CDataVector) = _check_live(_owner(x))
_check_live(t::CDataTable) = _check_live(getfield(t, :owner))

function _with_live(f::F, owner::CDataOwner) where {F}
    lock(owner.lock)
    try
        owner.released && throw(ArgumentError("Arrow C Data object has been released"))
        return f()
    finally
        unlock(owner.lock)
    end
end

_with_live(f::F, x::CDataVector) where {F} = _with_live(f, _owner(x))
_with_live(f::F, t::CDataTable) where {F} = _with_live(f, getfield(t, :owner))

validitybitmap(x::CDataNull) = nothing
nullcount(x::CDataNull) = x.len
nullcount(x::CDataVector) = validitybitmap(x).null_count
nullcount(x::CDataSlice) = count(i -> ismissing(x[i]), eachindex(x))
getmetadata(x::CDataSlice) = getmetadata(x.parent)
getmetadata(t::CDataTable) = getfield(t, :metadata)

@inline function _valid_bit(bytes::Vector{UInt8}, bitoffset::Int, i::Integer)
    pos = bitoffset + Int(i) - 1
    byte = @inbounds bytes[(pos >>> 3) + 1]
    return getbit(byte, (pos & 0x07) + 1)
end

@inline function _valid(v::CDataValidity, i::Integer)
    v.null_count == 0 && return true
    return _valid_bit(v.bytes, v.bitoffset, i)
end

function _count_nulls(v::CDataValidity)
    n = 0
    for i = 1:v.len
        n += _valid_bit(v.bytes, v.bitoffset, i) ? 0 : 1
    end
    return n
end

@propagate_inbounds function Base.getindex(x::CDataNull, i::Integer)
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        return missing
    end
end

@propagate_inbounds function Base.getindex(x::CDataPrimitive{T}, i::Integer) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        return @inbounds ArrowTypes.fromarrow(T, x.data[i])
    end
end

@propagate_inbounds function Base.getindex(x::CDataBool{T}, i::Integer) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        pos = x.bitoffset + Int(i) - 1
        byte = @inbounds x.data[(pos >>> 3) + 1]
        return ArrowTypes.fromarrow(T, getbit(byte, (pos & 0x07) + 1))
    end
end

@propagate_inbounds function Base.getindex(x::CDataBinary{T}, i::Integer) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        lo = Int(@inbounds x.offsets[i]) + 1
        hi = Int(@inbounds x.offsets[i + 1])
        n = hi - lo + 1
        if n == 0
            return ArrowTypes.fromarrow(T, "")
        end
        data = x.data
        owner = _owner(x)
        GC.@preserve x data owner begin
            return ArrowTypes.fromarrow(T, pointer(data, lo), n)
        end
    end
end

@propagate_inbounds function Base.getindex(
    x::CDataFixedSizeBinary{T},
    i::Integer,
) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        offset = (Int(i) - 1) * x.bytewidth
        tup = ntuple(j -> @inbounds(x.data[offset + j]), x.bytewidth)
        return ArrowTypes.fromarrow(T, tup)
    end
end

@propagate_inbounds function Base.getindex(x::CDataList{T}, i::Integer) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        lo = Int(@inbounds x.offsets[i]) + 1
        hi = Int(@inbounds x.offsets[i + 1])
        return ArrowTypes.fromarrow(T, @view x.data[lo:hi])
    end
end

@propagate_inbounds function Base.getindex(
    x::CDataFixedSizeList{T},
    i::Integer,
) where {T}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        offset = (x.offset + Int(i) - 1) * x.listsize
        tup = ntuple(j -> @inbounds(x.data[offset + j]), x.listsize)
        return ArrowTypes.fromarrow(T, tup)
    end
end

@propagate_inbounds function Base.getindex(
    x::CDataStruct{T,S,fnames},
    i::Integer,
) where {T,S,fnames}
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        if !_valid(x.validity, i)
            return missing
        end
        j = x.offset + Int(i)
        vals = ntuple(k -> @inbounds(x.data[k][j]), fieldcount(S))
        NT = Base.nonmissingtype(T)
        if isnamedtuple(NT) || istuple(NT)
            return ArrowTypes.fromarrow(T, NT(vals))
        else
            return ArrowTypes.fromarrow(T, _fromarrowstruct(NT, Val{fnames}(), vals...))
        end
    end
end

@propagate_inbounds function Base.getindex(x::CDataSlice, i::Integer)
    return _with_live(x) do
        @boundscheck checkbounds(x, i)
        return @inbounds x.parent[x.first + Int(i) - 1]
    end
end

function Base.collect(x::CDataVector{T}) where {T}
    return _with_live(x) do
        out = Vector{T}(undef, length(x))
        for i in eachindex(x)
            @inbounds out[i] = x[i]
        end
        return out
    end
end

Base.copy(x::CDataVector) = collect(x)

function Base.copy(t::CDataTable)
    return _with_live(t) do
        names = getfield(t, :names)
        columns = getfield(t, :columns)
        return NamedTuple{Tuple(names)}(Tuple(copy(col) for col in columns))
    end
end

Tables.istable(::Type{CDataTable}) = true
Tables.columnaccess(::Type{CDataTable}) = true
Tables.columns(t::CDataTable) = t
Tables.columnnames(t::CDataTable) = getfield(t, :names)
Tables.schema(t::CDataTable) = Tables.Schema(getfield(t, :names), getfield(t, :types))
Tables.getcolumn(t::CDataTable, i::Int) = (_check_live(t); getfield(t, :columns)[i])
Tables.getcolumn(t::CDataTable, nm::Symbol) = (_check_live(t); getfield(t, :lookup)[nm])
Tables.rowcount(t::CDataTable) = getfield(t, :rowcount)

Base.getindex(t::CDataTable, i::Int) = Tables.getcolumn(t, i)
Base.getindex(t::CDataTable, nm::Symbol) = Tables.getcolumn(t, nm)
function Base.getproperty(t::CDataTable, nm::Symbol)
    lookup = getfield(t, :lookup)
    haskey(lookup, nm) && return Tables.getcolumn(t, nm)
    return getfield(t, nm)
end
Base.propertynames(t::CDataTable, private::Bool=false) =
    private ? fieldnames(typeof(t)) : Tuple(getfield(t, :names))

DataAPI.metadatasupport(::Type{CDataTable}) = (read=true, write=false)
DataAPI.colmetadatasupport(::Type{CDataTable}) = (read=true, write=false)

function DataAPI.metadata(t::CDataTable; style::Bool=false)
    meta = getmetadata(t)
    meta === nothing && return Dict{String,Any}()
    if style
        return Dict(k => (v, :default) for (k, v) in meta)
    else
        return Dict(meta)
    end
end

function DataAPI.metadata(t::CDataTable, key::AbstractString; style::Bool=false)
    meta = getmetadata(t)
    meta === nothing && throw(KeyError(key))
    val = meta[key]
    return style ? (val, :default) : val
end

function DataAPI.metadata(t::CDataTable, key::AbstractString, default; style::Bool=false)
    meta = getmetadata(t)
    if meta !== nothing && haskey(meta, key)
        val = meta[key]
        return style ? (val, :default) : val
    end
    return style ? (default, :default) : default
end

function DataAPI.metadatakeys(t::CDataTable)
    meta = getmetadata(t)
    meta === nothing && return ()
    return keys(meta)
end

function DataAPI.colmetadata(t::CDataTable; style::Bool=false)
    pairs = Pair{Symbol,Any}[]
    for col in getfield(t, :names)
        meta = getmetadata(t[col])
        if meta !== nothing && !isempty(meta)
            push!(pairs, col => DataAPI.colmetadata(t, col; style=style))
        end
    end
    return Dict(pairs)
end

function DataAPI.colmetadata(t::CDataTable, col; style::Bool=false)
    meta = getmetadata(t[col])
    meta === nothing && return Dict{String,Any}()
    if style
        return Dict(k => (v, :default) for (k, v) in meta)
    else
        return Dict(meta)
    end
end

function DataAPI.colmetadata(
    t::CDataTable,
    col,
    key::AbstractString;
    style::Bool=false,
)
    meta = getmetadata(t[col])
    meta === nothing && throw(KeyError(key))
    val = meta[key]
    return style ? (val, :default) : val
end

function DataAPI.colmetadata(
    t::CDataTable,
    col,
    key::AbstractString,
    default;
    style::Bool=false,
)
    meta = getmetadata(t[col])
    if meta !== nothing && haskey(meta, key)
        val = meta[key]
        return style ? (val, :default) : val
    end
    return style ? (default, :default) : default
end

function DataAPI.colmetadatakeys(t::CDataTable, col)
    meta = getmetadata(t[col])
    meta === nothing && return ()
    return keys(meta)
end

function DataAPI.colmetadatakeys(t::CDataTable)
    return (
        col => DataAPI.colmetadatakeys(t, col) for
        col in Tables.columnnames(t) if getmetadata(t[col]) !== nothing
    )
end

function release_c_data(owner::CDataOwner)
    lock(owner.lock)
    try
        owner.released && return
        owner.released = true
        array_ptr = owner.array_ptr
        schema_ptr = owner.schema_ptr
        if array_ptr != C_NULL
            array = unsafe_load(array_ptr)
            if array.release != C_NULL
                ccall(array.release, Cvoid, (Ptr{ArrowArray},), array_ptr)
            end
        end
        if schema_ptr != C_NULL
            schema = unsafe_load(schema_ptr)
            if schema.release != C_NULL
                ccall(schema.release, Cvoid, (Ptr{ArrowSchema},), schema_ptr)
            end
        end
        return
    finally
        unlock(owner.lock)
    end
end

"""
    Arrow.release_c_data(x)

Release C Data resources owned by an imported array or table. The call is
idempotent. Reads through imported arrays throw after release.
"""
release_c_data(x::CDataVector) = release_c_data(_owner(x))
release_c_data(t::CDataTable) = release_c_data(getfield(t, :owner))

function _to_int(x::Int64, name)
    x > typemax(Int) && throw(ArgumentError("$name exceeds the Julia Int range"))
    return Int(x)
end

function _checked_nonnegative(x::Int64, name)
    x < 0 && throw(ArgumentError("$name must be nonnegative"))
    return _to_int(x, name)
end

function _checked_add(a::Int, b::Int, name)
    b > typemax(Int) - a && throw(ArgumentError("$name overflows"))
    return a + b
end

function _checked_mul(a::Int, b::Int, name)
    a != 0 && b > typemax(Int) ÷ a && throw(ArgumentError("$name overflows"))
    return a * b
end

function _parse_c_data_format(format::AbstractString)
    format == "n" && return CDataNullFormat()
    format == "b" && return CDataBoolFormat()
    format == "c" && return CDataPrimitiveFormat(Int8)
    format == "C" && return CDataPrimitiveFormat(UInt8)
    format == "s" && return CDataPrimitiveFormat(Int16)
    format == "S" && return CDataPrimitiveFormat(UInt16)
    format == "i" && return CDataPrimitiveFormat(Int32)
    format == "I" && return CDataPrimitiveFormat(UInt32)
    format == "l" && return CDataPrimitiveFormat(Int64)
    format == "L" && return CDataPrimitiveFormat(UInt64)
    format == "e" && return CDataPrimitiveFormat(Float16)
    format == "f" && return CDataPrimitiveFormat(Float32)
    format == "g" && return CDataPrimitiveFormat(Float64)
    format == "z" && return CDataBinaryFormat{Int32}(Base.CodeUnits)
    format == "Z" && return CDataBinaryFormat{Int64}(Base.CodeUnits)
    format == "u" && return CDataBinaryFormat{Int32}(String)
    format == "U" && return CDataBinaryFormat{Int64}(String)
    format == "+l" && return CDataListFormat{Int32}()
    format == "+L" && return CDataListFormat{Int64}()
    format == "+s" && return CDataStructFormat()
    format == "tdD" && return CDataPrimitiveFormat(Date{Meta.DateUnit.DAY,Int32})
    format == "tdm" && return CDataPrimitiveFormat(Date{Meta.DateUnit.MILLISECOND,Int64})
    format == "tts" && return CDataPrimitiveFormat(Time{Meta.TimeUnit.SECOND,Int32})
    format == "ttm" && return CDataPrimitiveFormat(Time{Meta.TimeUnit.MILLISECOND,Int32})
    format == "ttu" && return CDataPrimitiveFormat(Time{Meta.TimeUnit.MICROSECOND,Int64})
    format == "ttn" && return CDataPrimitiveFormat(Time{Meta.TimeUnit.NANOSECOND,Int64})
    format == "tDs" && return CDataPrimitiveFormat(Duration{Meta.TimeUnit.SECOND})
    format == "tDm" && return CDataPrimitiveFormat(Duration{Meta.TimeUnit.MILLISECOND})
    format == "tDu" && return CDataPrimitiveFormat(Duration{Meta.TimeUnit.MICROSECOND})
    format == "tDn" && return CDataPrimitiveFormat(Duration{Meta.TimeUnit.NANOSECOND})
    format == "tiM" && return CDataPrimitiveFormat(Interval{Meta.IntervalUnit.YEAR_MONTH,Int32})
    format == "tiD" && return CDataPrimitiveFormat(Interval{Meta.IntervalUnit.DAY_TIME,Int64})
    if startswith(format, "ts")
        return CDataPrimitiveFormat(_parse_timestamp_format(format))
    elseif startswith(format, "d:")
        return CDataPrimitiveFormat(_parse_decimal_format(format))
    elseif startswith(format, "w:")
        return CDataFixedSizeBinaryFormat(
            _parse_positive_int(format[3:end], format, _CDATA_MAX_FIXED_SIZE),
        )
    elseif startswith(format, "+w:")
        return CDataFixedSizeListFormat(
            _parse_positive_int(format[4:end], format, _CDATA_MAX_FIXED_SIZE),
        )
    end
    throw(ArgumentError("unsupported Arrow C Data format string: $format"))
end

function _parse_positive_int(s, format, max_value::Int=typemax(Int))
    n = try
        parse(Int, s)
    catch
        throw(ArgumentError("invalid Arrow C Data format string: $format"))
    end
    n <= 0 && throw(ArgumentError("invalid Arrow C Data format string: $format"))
    n > max_value &&
        throw(ArgumentError("Arrow C Data format size exceeds the import limit: $format"))
    return n
end

function _parse_timestamp_format(format)
    length(format) >= 4 || throw(ArgumentError("invalid Arrow C Data format string: $format"))
    format[4] == ':' || throw(ArgumentError("invalid Arrow C Data format string: $format"))
    unit = format[3]
    U =
        unit == 's' ? Meta.TimeUnit.SECOND :
        unit == 'm' ? Meta.TimeUnit.MILLISECOND :
        unit == 'u' ? Meta.TimeUnit.MICROSECOND :
        unit == 'n' ? Meta.TimeUnit.NANOSECOND :
        throw(ArgumentError("invalid Arrow C Data timestamp unit: $format"))
    tz = length(format) == 4 ? nothing : Symbol(format[5:end])
    return Timestamp{U,tz}
end

function _parse_decimal_format(format)
    parts = split(format[3:end], ',')
    2 <= length(parts) <= 3 ||
        throw(ArgumentError("invalid Arrow C Data decimal format: $format"))
    precision = parse(Int, parts[1])
    scale = parse(Int, parts[2])
    bitwidth = length(parts) == 3 ? parse(Int, parts[3]) : 128
    precision > 0 || throw(ArgumentError("decimal precision must be positive"))
    if bitwidth == 128
        return Decimal{precision,scale,Int128}
    elseif bitwidth == 256
        return Decimal{precision,scale,Int256}
    else
        throw(ArgumentError("unsupported decimal bit width: $bitwidth"))
    end
end

_expected_buffers(::CDataNullFormat) = 0
_expected_buffers(::CDataBoolFormat) = 2
_expected_buffers(::CDataPrimitiveFormat) = 2
_expected_buffers(::CDataBinaryFormat) = 3
_expected_buffers(::CDataFixedSizeBinaryFormat) = 2
_expected_buffers(::CDataListFormat) = 2
_expected_buffers(::CDataFixedSizeListFormat) = 1
_expected_buffers(::CDataStructFormat) = 1

_expected_children(::CDataNullFormat) = 0
_expected_children(::CDataBoolFormat) = 0
_expected_children(::CDataPrimitiveFormat) = 0
_expected_children(::CDataBinaryFormat) = 0
_expected_children(::CDataFixedSizeBinaryFormat) = 0
_expected_children(::CDataListFormat) = 1
_expected_children(::CDataFixedSizeListFormat) = 1
_expected_children(::CDataStructFormat) = nothing

function _nullable(schema::ArrowSchema, null_count::Int)
    return (schema.flags & ARROW_FLAG_NULLABLE) != 0 || null_count != 0
end

function _julia_type(storage::Type, nullable::Bool, convert::Bool)
    T = convert ? finaljuliatype(storage) : storage
    return nullable ? Union{T,Missing} : T
end

function _load_name(ptr::Cstring)
    ptr == C_NULL && return nothing
    name = _unsafe_string_bounded(ptr, _CDATA_MAX_NAME_BYTES, "ArrowSchema.name")
    return isempty(name) ? nothing : name
end

function _unsafe_string_bounded(ptr::Cstring, maxbytes::Int, name)
    bytes = UInt8[]
    sizehint!(bytes, min(maxbytes, 128))
    p = Ptr{UInt8}(ptr)
    for i = 1:maxbytes
        byte = unsafe_load(p, i)
        byte == 0x00 && return String(bytes)
        push!(bytes, byte)
    end
    throw(ArgumentError("$name exceeds the import limit"))
end

function _metadata_dict(pairs)
    isempty(pairs) && return Base.ImmutableDict{String,String}()
    return toidict(pairs)
end

@inline function _unsafe_load_int32(p::Ptr{UInt8})
    b1 = UInt32(unsafe_load(p, 1))
    b2 = UInt32(unsafe_load(p, 2))
    b3 = UInt32(unsafe_load(p, 3))
    b4 = UInt32(unsafe_load(p, 4))
    u =
        ENDIAN_BOM == 0x04030201 ? b1 | (b2 << 8) | (b3 << 16) | (b4 << 24) :
        ENDIAN_BOM == 0x01020304 ? (b1 << 24) | (b2 << 16) | (b3 << 8) | b4 :
        error("unsupported host byte order")
    return reinterpret(Int32, u)
end

function _parse_c_metadata(ptr::Cstring)
    ptr == C_NULL && return nothing
    p = Ptr{UInt8}(ptr)
    count = Int(_unsafe_load_int32(p))
    count < 0 && throw(ArgumentError("Arrow C Data metadata pair count is negative"))
    count > _CDATA_MAX_METADATA_PAIRS &&
        throw(ArgumentError("Arrow C Data metadata pair count exceeds the limit"))
    pos = 4
    total = 4
    pairs = Pair{String,String}[]
    for _ = 1:count
        key_len = Int(_unsafe_load_int32(p + pos))
        pos += 4
        total += 4
        key_len < 0 && throw(ArgumentError("Arrow C Data metadata key length is negative"))
        key_len > _CDATA_MAX_METADATA_FIELD_BYTES &&
            throw(ArgumentError("Arrow C Data metadata key length exceeds the limit"))
        total = _checked_add(total, key_len, "metadata byte count")
        total > _CDATA_MAX_METADATA_BYTES &&
            throw(ArgumentError("Arrow C Data metadata byte count exceeds the limit"))
        key = unsafe_string(p + pos, key_len)
        pos += key_len

        value_len = Int(_unsafe_load_int32(p + pos))
        pos += 4
        total += 4
        value_len < 0 && throw(ArgumentError("Arrow C Data metadata value length is negative"))
        value_len > _CDATA_MAX_METADATA_FIELD_BYTES &&
            throw(ArgumentError("Arrow C Data metadata value length exceeds the limit"))
        total = _checked_add(total, value_len, "metadata byte count")
        total > _CDATA_MAX_METADATA_BYTES &&
            throw(ArgumentError("Arrow C Data metadata byte count exceeds the limit"))
        value = unsafe_string(p + pos, value_len)
        pos += value_len
        push!(pairs, key => value)
    end
    return _metadata_dict(pairs)
end

function _load_buffers(array::ArrowArray, expected::Int)
    n_buffers = _checked_nonnegative(array.n_buffers, "ArrowArray.n_buffers")
    n_buffers > _CDATA_MAX_BUFFERS &&
        throw(ArgumentError("ArrowArray.n_buffers exceeds the import limit"))
    n_buffers >= expected ||
        throw(ArgumentError("ArrowArray has fewer buffers than required"))
    if expected > 0 && array.buffers == C_NULL
        throw(ArgumentError("ArrowArray.buffers is NULL"))
    end
    buffers = Ptr{Cvoid}[]
    for i = 1:expected
        push!(buffers, unsafe_load(array.buffers, i))
    end
    return buffers
end

function _load_child_ptrs(schema::ArrowSchema, array::ArrowArray, count::Int)
    count == 0 && return Ptr{ArrowSchema}[], Ptr{ArrowArray}[]
    schema.children == C_NULL && throw(ArgumentError("ArrowSchema.children is NULL"))
    array.children == C_NULL && throw(ArgumentError("ArrowArray.children is NULL"))
    schema_children = Ptr{ArrowSchema}[]
    array_children = Ptr{ArrowArray}[]
    for i = 1:count
        schema_child = unsafe_load(schema.children, i)
        array_child = unsafe_load(array.children, i)
        schema_child == C_NULL && throw(ArgumentError("ArrowSchema child pointer is NULL"))
        array_child == C_NULL && throw(ArgumentError("ArrowArray child pointer is NULL"))
        push!(schema_children, schema_child)
        push!(array_children, array_child)
    end
    return schema_children, array_children
end

function _validate_common(schema::ArrowSchema, array::ArrowArray, top_level::Bool)
    schema.format == C_NULL && throw(ArgumentError("ArrowSchema.format is NULL"))
    if top_level
        schema.release == C_NULL && throw(ArgumentError("ArrowSchema.release is NULL"))
        array.release == C_NULL && throw(ArgumentError("ArrowArray.release is NULL"))
    elseif schema.release == C_NULL || array.release == C_NULL
        throw(ArgumentError("released Arrow C Data child structure"))
    end
    len = _checked_nonnegative(array.length, "ArrowArray.length")
    offset = _checked_nonnegative(array.offset, "ArrowArray.offset")
    _checked_nonnegative(array.n_buffers, "ArrowArray.n_buffers")
    n_children = _checked_nonnegative(array.n_children, "ArrowArray.n_children")
    n_children > _CDATA_MAX_CHILDREN &&
        throw(ArgumentError("ArrowArray.n_children exceeds the import limit"))
    schema_children = _checked_nonnegative(schema.n_children, "ArrowSchema.n_children")
    schema_children > _CDATA_MAX_CHILDREN &&
        throw(ArgumentError("ArrowSchema.n_children exceeds the import limit"))
    schema_children == n_children ||
        throw(ArgumentError("ArrowSchema and ArrowArray child counts differ"))
    null_count = array.null_count
    if !(null_count == -1 || 0 <= null_count <= array.length)
        throw(ArgumentError("ArrowArray.null_count is out of range"))
    end
    if (schema.dictionary == C_NULL) != (array.dictionary == C_NULL)
        throw(ArgumentError("ArrowSchema and ArrowArray dictionary pointers differ"))
    end
    schema.dictionary == C_NULL ||
        throw(ArgumentError("dictionary encoded Arrow C Data import is not supported"))
    return len, offset, Int(null_count), n_children
end

function _validate_node(
    schema_ptr::Ptr{ArrowSchema},
    array_ptr::Ptr{ArrowArray};
    top_level::Bool=false,
    depth::Int=1,
)
    depth > _CDATA_MAX_DEPTH &&
        throw(ArgumentError("Arrow C Data nesting exceeds the import limit"))
    schema_ptr == C_NULL && throw(ArgumentError("ArrowSchema pointer is NULL"))
    array_ptr == C_NULL && throw(ArgumentError("ArrowArray pointer is NULL"))
    schema = unsafe_load(schema_ptr)
    array = unsafe_load(array_ptr)
    len, offset, null_count, n_children = _validate_common(schema, array, top_level)
    format = _parse_c_data_format(
        _unsafe_string_bounded(schema.format, _CDATA_MAX_FORMAT_BYTES, "ArrowSchema.format"),
    )
    expected_children = _expected_children(format)
    if expected_children !== nothing && n_children != expected_children
        throw(ArgumentError("Arrow C Data child count does not match the format"))
    end
    buffers = _load_buffers(array, _expected_buffers(format))
    schema_child_ptrs, array_child_ptrs = _load_child_ptrs(schema, array, n_children)
    children = CDataNode[]
    for i in eachindex(schema_child_ptrs)
        push!(
            children,
            _validate_node(
                schema_child_ptrs[i],
                array_child_ptrs[i];
                top_level=false,
                depth=depth + 1,
            ),
        )
    end
    node = CDataNode(
        schema_ptr,
        array_ptr,
        schema,
        array,
        format,
        _load_name(schema.name),
        _parse_c_metadata(schema.metadata),
        buffers,
        children,
        len,
        offset,
        null_count,
    )
    _validate_layout(node)
    return node
end

function _validate_layout(node::CDataNode)
    total = _checked_add(node.offset, node.len, "ArrowArray offset plus length")
    validity_bytes = cld(total, 8)
    if _expected_buffers(node.format) > 0
        validity = node.buffers[1]
        if node.null_count == -1
            if node.len > 0 && validity == C_NULL
                throw(ArgumentError("unknown null count requires a validity bitmap"))
            end
        elseif node.null_count > 0 && validity == C_NULL
            throw(ArgumentError("null values require a validity bitmap"))
        end
        validity_bytes >= 0 || throw(ArgumentError("invalid validity bitmap size"))
    end
    _validate_data_layout(node.format, node, total)
    return
end

function _validate_data_layout(::CDataNullFormat, node::CDataNode, total::Int)
    return
end

function _validate_data_layout(::CDataBoolFormat, node::CDataNode, total::Int)
    nbytes = cld(total, 8)
    nbytes > 0 && node.buffers[2] == C_NULL &&
        throw(ArgumentError("boolean data buffer is NULL"))
    return
end

function _validate_data_layout(format::CDataPrimitiveFormat, node::CDataNode, total::Int)
    nbytes = _checked_mul(total, sizeof(format.storage), "primitive data byte count")
    nbytes > 0 && node.buffers[2] == C_NULL &&
        throw(ArgumentError("primitive data buffer is NULL"))
    return
end

function _validate_data_layout(
    format::CDataFixedSizeBinaryFormat,
    node::CDataNode,
    total::Int,
)
    nbytes = _checked_mul(total, format.bytewidth, "fixed size binary byte count")
    nbytes > 0 && node.buffers[2] == C_NULL &&
        throw(ArgumentError("fixed size binary data buffer is NULL"))
    return
end

function _validate_data_layout(format::CDataBinaryFormat{O}, node::CDataNode, total::Int) where {O}
    node.buffers[2] == C_NULL && throw(ArgumentError("offset buffer is NULL"))
    first, last = _validate_offsets(Ptr{O}(node.buffers[2]), node.offset, node.len)
    last < first && throw(ArgumentError("offsets are not monotonic"))
    last > first && node.buffers[3] == C_NULL &&
        throw(ArgumentError("data buffer is NULL"))
    return
end

function _validate_data_layout(format::CDataListFormat{O}, node::CDataNode, total::Int) where {O}
    node.buffers[2] == C_NULL && throw(ArgumentError("offset buffer is NULL"))
    first, last = _validate_offsets(Ptr{O}(node.buffers[2]), node.offset, node.len)
    last < first && throw(ArgumentError("offsets are not monotonic"))
    child = node.children[1]
    last <= child.len ||
        throw(ArgumentError("list offset exceeds child length"))
    return
end

function _validate_data_layout(
    format::CDataFixedSizeListFormat,
    node::CDataNode,
    total::Int,
)
    required = _checked_mul(total, format.listsize, "fixed size list child length")
    node.children[1].len >= required ||
        throw(ArgumentError("fixed size list child is too short"))
    return
end

function _validate_data_layout(::CDataStructFormat, node::CDataNode, total::Int)
    for child in node.children
        child.len >= total ||
            throw(ArgumentError("struct child is too short"))
    end
    return
end

function _validate_offsets(ptr::Ptr{O}, offset::Int, len::Int) where {O}
    n = _checked_add(_checked_add(offset, len, "offset count"), 1, "offset count")
    _checked_mul(n, sizeof(O), "offset buffer byte count")
    _checked_mul(offset, sizeof(O), "offset buffer byte offset")
    offsets = unsafe_wrap(Array, ptr, n; own=false)
    first = _offset_to_int(offsets[offset + 1], "offset")
    prev = first
    for i = (offset + 2):n
        cur = _offset_to_int(offsets[i], "offset")
        cur < prev && throw(ArgumentError("offsets are not monotonic"))
        prev = cur
    end
    return first, prev
end

function _offset_to_int(x, name)
    x < 0 && throw(ArgumentError("$name is negative"))
    x > typemax(Int) && throw(ArgumentError("$name exceeds the Julia Int range"))
    return Int(x)
end

function _make_validity(node::CDataNode)
    if _expected_buffers(node.format) == 0
        return CDataValidity(UInt8[], 0, node.len, 0)
    end
    ptr = node.buffers[1]
    if node.null_count == 0 || node.len == 0
        return CDataValidity(UInt8[], 0, node.len, 0)
    end
    nbytes = cld(_checked_add(node.offset, node.len, "validity bitmap length"), 8)
    bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), nbytes; own=false)
    validity = CDataValidity(bytes, node.offset, node.len, max(node.null_count, 0))
    if node.null_count == -1
        return CDataValidity(bytes, node.offset, node.len, _count_nulls(validity))
    else
        return validity
    end
end

function _wrap_data(ptr::Ptr{Cvoid}, ::Type{T}, offset::Int, len::Int) where {T}
    len == 0 && return T[]
    p = Ptr{T}(ptr) + _checked_mul(offset, sizeof(T), "data buffer byte offset")
    return unsafe_wrap(Array, p, len; own=false)
end

function _import_node(node::CDataNode, owner::CDataOwner, convert::Bool)
    return _import_node(node.format, node, owner, convert)
end

function _import_node(::CDataNullFormat, node::CDataNode, owner::CDataOwner, convert::Bool)
    return CDataNull{Missing}(owner, node.len, node.metadata)
end

function _import_node(format::CDataPrimitiveFormat, node::CDataNode, owner::CDataOwner, convert::Bool)
    validity = _make_validity(node)
    nullable = _nullable(node.schema, validity.null_count)
    T = _julia_type(format.storage, nullable, convert)
    data = _wrap_data(node.buffers[2], format.storage, node.offset, node.len)
    return CDataPrimitive{T,format.storage,typeof(data)}(owner, validity, data, node.metadata)
end

function _import_node(::CDataBoolFormat, node::CDataNode, owner::CDataOwner, convert::Bool)
    validity = _make_validity(node)
    nullable = _nullable(node.schema, validity.null_count)
    T = nullable ? Union{Bool,Missing} : Bool
    nbytes = cld(_checked_add(node.offset, node.len, "boolean bitmap length"), 8)
    data = nbytes == 0 ? UInt8[] : unsafe_wrap(Array, Ptr{UInt8}(node.buffers[2]), nbytes; own=false)
    return CDataBool{T}(owner, validity, data, node.offset, node.len, node.metadata)
end

function _import_node(
    format::CDataBinaryFormat{O},
    node::CDataNode,
    owner::CDataOwner,
    convert::Bool,
) where {O}
    validity = _make_validity(node)
    nullable = _nullable(node.schema, validity.null_count)
    T = nullable ? Union{format.juliatype,Missing} : format.juliatype
    p =
        Ptr{O}(node.buffers[2]) +
        _checked_mul(node.offset, sizeof(O), "offset buffer byte offset")
    offsets = unsafe_wrap(Array, p, node.len + 1; own=false)
    data_len = offsets[end] == offsets[1] ? 0 : Int(offsets[end])
    data =
        data_len == 0 ? UInt8[] :
        unsafe_wrap(Array, Ptr{UInt8}(node.buffers[3]), data_len; own=false)
    return CDataBinary{T,O,typeof(offsets)}(owner, validity, offsets, data, node.len, node.metadata)
end

function _import_node(
    format::CDataFixedSizeBinaryFormat,
    node::CDataNode,
    owner::CDataOwner,
    convert::Bool,
)
    validity = _make_validity(node)
    nullable = _nullable(node.schema, validity.null_count)
    storage = NTuple{format.bytewidth,UInt8}
    T = nullable ? Union{storage,Missing} : storage
    nbytes = _checked_mul(node.len, format.bytewidth, "fixed size binary byte count")
    data =
        nbytes == 0 ? UInt8[] :
        unsafe_wrap(
            Array,
            Ptr{UInt8}(node.buffers[2]) +
            _checked_mul(node.offset, format.bytewidth, "fixed size binary byte offset"),
            nbytes;
            own=false,
        )
    return CDataFixedSizeBinary{T}(
        owner,
        validity,
        data,
        format.bytewidth,
        node.len,
        node.metadata,
    )
end

function _import_node(
    format::CDataListFormat{O},
    node::CDataNode,
    owner::CDataOwner,
    convert::Bool,
) where {O}
    validity = _make_validity(node)
    child = _import_node(node.children[1], owner, convert)
    nullable = _nullable(node.schema, validity.null_count)
    storage = Vector{eltype(child)}
    T = nullable ? Union{storage,Missing} : storage
    p =
        Ptr{O}(node.buffers[2]) +
        _checked_mul(node.offset, sizeof(O), "offset buffer byte offset")
    offsets = unsafe_wrap(Array, p, node.len + 1; own=false)
    return CDataList{T,O,typeof(offsets),typeof(child)}(
        owner,
        validity,
        offsets,
        child,
        node.len,
        node.metadata,
    )
end

function _import_node(
    format::CDataFixedSizeListFormat,
    node::CDataNode,
    owner::CDataOwner,
    convert::Bool,
)
    validity = _make_validity(node)
    child = _import_node(node.children[1], owner, convert)
    nullable = _nullable(node.schema, validity.null_count)
    storage = NTuple{format.listsize,eltype(child)}
    T = nullable ? Union{storage,Missing} : storage
    return CDataFixedSizeList{T,typeof(child)}(
        owner,
        validity,
        child,
        format.listsize,
        node.offset,
        node.len,
        node.metadata,
    )
end

function _import_node(::CDataStructFormat, node::CDataNode, owner::CDataOwner, convert::Bool)
    validity = _make_validity(node)
    children = Tuple(_import_node(child, owner, convert) for child in node.children)
    names = Tuple(
        Symbol(child.name === nothing ? "f$(i)" : child.name) for
        (i, child) in enumerate(node.children)
    )
    types = Tuple(eltype(child) for child in children)
    storage = NamedTuple{names,Tuple{types...}}
    nullable = _nullable(node.schema, validity.null_count)
    T = nullable ? Union{storage,Missing} : storage
    return CDataStruct{T,typeof(children),names}(
        owner,
        validity,
        children,
        node.offset,
        node.len,
        node.metadata,
    )
end

function _slice_for_table(child::CDataVector, offset::Int, len::Int)
    if offset == 0 && length(child) == len
        return child
    else
        first = offset + 1
        return CDataSlice{eltype(child),typeof(child)}(child, first, len)
    end
end

function _table_from_struct(node::CDataNode, owner::CDataOwner, convert::Bool)
    columns = AbstractVector[]
    names = Symbol[]
    types = Type[]
    for (i, child_node) in enumerate(node.children)
        child = _import_node(child_node, owner, convert)
        col = _slice_for_table(child, node.offset, node.len)
        name = Symbol(child_node.name === nothing ? "f$(i)" : child_node.name)
        push!(names, name)
        push!(types, eltype(col))
        push!(columns, col)
    end
    lookup = Dict{Symbol,AbstractVector}(names[i] => columns[i] for i in eachindex(names))
    return CDataTable(names, types, columns, lookup, node.metadata, owner, node.len)
end

"""
    Arrow.from_c_data(schema_ptr, array_ptr; convert=true)

Import an Arrow C Data Interface schema and array pair. Imported buffers are
viewed without copying and are released by `release_c_data` or finalization.
Use `copy` or `collect` on imported arrays to make Julia owned arrays.

A top level struct array is returned as a Tables.jl column table.
"""
function from_c_data(
    schema_ptr::Ptr{ArrowSchema},
    array_ptr::Ptr{ArrowArray};
    convert::Bool=true,
)
    owner = CDataOwner(schema_ptr, array_ptr)
    try
        node = _validate_node(schema_ptr, array_ptr; top_level=true)
        if node.format isa CDataStructFormat
            return _table_from_struct(node, owner, convert)
        else
            return _import_node(node, owner, convert)
        end
    catch
        release_c_data(owner)
        rethrow()
    end
end

from_c_data(schema_ptr::Ptr{Cvoid}, array_ptr::Ptr{Cvoid}; kw...) =
    from_c_data(Ptr{ArrowSchema}(schema_ptr), Ptr{ArrowArray}(array_ptr); kw...)
