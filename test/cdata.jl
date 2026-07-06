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

function _cdata_release_schema(ptr::Ptr{Arrow.ArrowSchema})
    schema = unsafe_load(ptr)
    unsafe_store!(
        ptr,
        Arrow.ArrowSchema(
            schema.format,
            schema.name,
            schema.metadata,
            schema.flags,
            schema.n_children,
            schema.children,
            schema.dictionary,
            C_NULL,
            schema.private_data,
        ),
    )
    return
end

function _cdata_release_array(ptr::Ptr{Arrow.ArrowArray})
    array = unsafe_load(ptr)
    unsafe_store!(
        ptr,
        Arrow.ArrowArray(
            array.length,
            array.null_count,
            array.offset,
            array.n_buffers,
            array.n_children,
            array.buffers,
            array.children,
            array.dictionary,
            C_NULL,
            array.private_data,
        ),
    )
    return
end

const _CDATA_RELEASE_SCHEMA =
    @cfunction(_cdata_release_schema, Cvoid, (Ptr{Arrow.ArrowSchema},))
const _CDATA_RELEASE_ARRAY =
    @cfunction(_cdata_release_array, Cvoid, (Ptr{Arrow.ArrowArray},))

mutable struct CDataFixture
    schema::Ref{Arrow.ArrowSchema}
    array::Ref{Arrow.ArrowArray}
    roots::Vector{Any}
end

const _CDATA_FIXTURE_ROOTS = Any[]

_schema_ptr(x::CDataFixture) = Base.unsafe_convert(Ptr{Arrow.ArrowSchema}, x.schema)
_array_ptr(x::CDataFixture) = Base.unsafe_convert(Ptr{Arrow.ArrowArray}, x.array)

function _cstring_root(s::Union{Nothing,String}, roots)
    s === nothing && return Cstring(C_NULL)
    bytes = Vector{UInt8}(s * "\0")
    push!(roots, bytes)
    return Cstring(pointer(bytes))
end

function _metadata_root(meta::Union{Nothing,AbstractDict}, roots)
    meta === nothing && return Cstring(C_NULL)
    io = IOBuffer()
    write(io, Int32(length(meta)))
    for (k, v) in meta
        kb = codeunits(String(k))
        vb = codeunits(String(v))
        write(io, Int32(length(kb)))
        write(io, kb)
        write(io, Int32(length(vb)))
        write(io, vb)
    end
    bytes = take!(io)
    push!(roots, bytes)
    return Cstring(pointer(bytes))
end

function _cdata_fixture(
    fmt::String,
    len::Integer,
    buffers::Vector{Ptr{Cvoid}};
    name=nothing,
    metadata=nothing,
    flags::Int64=0,
    null_count::Int64=0,
    offset::Int64=0,
    children::Vector{CDataFixture}=CDataFixture[],
)
    roots = Any[]
    append!(roots, children)
    schema_ptrs = [Base.unsafe_convert(Ptr{Arrow.ArrowSchema}, child.schema) for child in children]
    array_ptrs = [Base.unsafe_convert(Ptr{Arrow.ArrowArray}, child.array) for child in children]
    if !isempty(children)
        push!(roots, schema_ptrs)
        push!(roots, array_ptrs)
    end
    buffer_ptrs = copy(buffers)
    if !isempty(buffer_ptrs)
        push!(roots, buffer_ptrs)
    end
    schema = Ref(
        Arrow.ArrowSchema(
            _cstring_root(fmt, roots),
            _cstring_root(name, roots),
            _metadata_root(metadata, roots),
            flags,
            Int64(length(children)),
            isempty(children) ? Ptr{Ptr{Arrow.ArrowSchema}}(C_NULL) :
            Ptr{Ptr{Arrow.ArrowSchema}}(pointer(schema_ptrs)),
            Ptr{Arrow.ArrowSchema}(C_NULL),
            _CDATA_RELEASE_SCHEMA,
            Ptr{Cvoid}(C_NULL),
        ),
    )
    array = Ref(
        Arrow.ArrowArray(
            Int64(len),
            null_count,
            offset,
            Int64(length(buffer_ptrs)),
            Int64(length(children)),
            isempty(buffer_ptrs) ? Ptr{Ptr{Cvoid}}(C_NULL) :
            Ptr{Ptr{Cvoid}}(pointer(buffer_ptrs)),
            isempty(children) ? Ptr{Ptr{Arrow.ArrowArray}}(C_NULL) :
            Ptr{Ptr{Arrow.ArrowArray}}(pointer(array_ptrs)),
            Ptr{Arrow.ArrowArray}(C_NULL),
            _CDATA_RELEASE_ARRAY,
            Ptr{Cvoid}(C_NULL),
        ),
    )
    push!(roots, schema)
    push!(roots, array)
    fixture = CDataFixture(schema, array, roots)
    push!(_CDATA_FIXTURE_ROOTS, fixture)
    return fixture
end

function _primitive_fixture(
    fmt,
    data::Vector{T};
    validity=nothing,
    null_count::Int64=0,
    flags::Int64=0,
    offset::Int64=0,
    len::Int=length(data) - Int(offset),
    name=nothing,
    metadata=nothing,
) where {T}
    roots = Any[data]
    buffers = Ptr{Cvoid}[
        validity === nothing ? Ptr{Cvoid}(C_NULL) : Ptr{Cvoid}(pointer(validity)),
        isempty(data) ? Ptr{Cvoid}(C_NULL) : Ptr{Cvoid}(pointer(data)),
    ]
    validity !== nothing && push!(roots, validity)
    fixture = _cdata_fixture(
        fmt,
        len,
        buffers;
        name=name,
        metadata=metadata,
        flags=flags,
        null_count=null_count,
        offset=offset,
    )
    append!(fixture.roots, roots)
    return fixture
end

@testset "Arrow C Data Interface import" begin
    @testset "ABI layout" begin
        ptr = sizeof(Ptr{Cvoid})
        @test sizeof(Arrow.ArrowSchema) == 7 * ptr + 2 * sizeof(Int64)
        @test sizeof(Arrow.ArrowArray) == 5 * ptr + 5 * sizeof(Int64)

        cc = Sys.which("cc")
        if cc === nothing
            @test_skip "C compiler not available"
        else
            c_src = """
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>
            struct ArrowSchema {
              const char* format;
              const char* name;
              const char* metadata;
              int64_t flags;
              int64_t n_children;
              struct ArrowSchema** children;
              struct ArrowSchema* dictionary;
              void (*release)(struct ArrowSchema*);
              void* private_data;
            };
            struct ArrowArray {
              int64_t length;
              int64_t null_count;
              int64_t offset;
              int64_t n_buffers;
              int64_t n_children;
              const void** buffers;
              struct ArrowArray** children;
              struct ArrowArray* dictionary;
              void (*release)(struct ArrowArray*);
              void* private_data;
            };
            int main(void) {
              printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu\\n",
                offsetof(struct ArrowSchema, format),
                offsetof(struct ArrowSchema, name),
                offsetof(struct ArrowSchema, metadata),
                offsetof(struct ArrowSchema, flags),
                offsetof(struct ArrowSchema, n_children),
                offsetof(struct ArrowSchema, children),
                offsetof(struct ArrowSchema, dictionary),
                offsetof(struct ArrowSchema, release),
                offsetof(struct ArrowSchema, private_data));
              printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\\n",
                offsetof(struct ArrowArray, length),
                offsetof(struct ArrowArray, null_count),
                offsetof(struct ArrowArray, offset),
                offsetof(struct ArrowArray, n_buffers),
                offsetof(struct ArrowArray, n_children),
                offsetof(struct ArrowArray, buffers),
                offsetof(struct ArrowArray, children),
                offsetof(struct ArrowArray, dictionary),
                offsetof(struct ArrowArray, release),
                offsetof(struct ArrowArray, private_data));
              return 0;
            }
            """
            mktempdir() do dir
                src = joinpath(dir, "layout.c")
                bin = joinpath(dir, "layout")
                write(src, c_src)
                run(`$cc -o $bin $src`)
                lines = split(readchomp(`$bin`), '\n')
                schema_offsets = parse.(Int, split(lines[1]))
                array_offsets = parse.(Int, split(lines[2]))
                for (i, offset) in enumerate(schema_offsets)
                    @test fieldoffset(Arrow.ArrowSchema, i) == offset
                end
                for (i, offset) in enumerate(array_offsets)
                    @test fieldoffset(Arrow.ArrowArray, i) == offset
                end
            end
        end
    end

    @testset "primitive arrays" begin
        f = _primitive_fixture("i", Int32[1, 2, 3])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test x isa Arrow.CDataVector
        @test collect(x) == Int32[1, 2, 3]
        @test copy(x) == Int32[1, 2, 3]

        validity = UInt8[0b00000101]
        f = _primitive_fixture(
            "i",
            Int32[10, 20, 30];
            validity=validity,
            null_count=Int64(1),
            flags=Arrow.ARROW_FLAG_NULLABLE,
        )
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test isequal(collect(x), Union{Int32,Missing}[10, missing, 30])

        f = _primitive_fixture(
            "i",
            Int32[10, 20, 30];
            validity=validity,
            null_count=Int64(-1),
            flags=Arrow.ARROW_FLAG_NULLABLE,
        )
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test Arrow.nullcount(x) == 1
        @test isequal(collect(x), Union{Int32,Missing}[10, missing, 30])
    end

    @testset "bool bit offsets" begin
        data = UInt8[0b10110110]
        validity = UInt8[0xff]
        f = _cdata_fixture(
            "b",
            5,
            Ptr{Cvoid}[Ptr{Cvoid}(pointer(validity)), Ptr{Cvoid}(pointer(data))];
            offset=Int64(3),
        )
        append!(f.roots, Any[data, validity])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == [false, true, true, false, true]
    end

    @testset "string and binary arrays" begin
        offsets = Int32[0, 3, 3, 6]
        bytes = Vector{UInt8}(codeunits("abcdef"))
        f = _cdata_fixture(
            "u",
            3,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets)), Ptr{Cvoid}(pointer(bytes))],
        )
        append!(f.roots, Any[offsets, bytes])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == ["abc", "", "def"]

        offsets64 = Int64[0, 2, 5]
        bytes2 = Vector{UInt8}(codeunits("hello"))
        f = _cdata_fixture(
            "U",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets64)), Ptr{Cvoid}(pointer(bytes2))],
        )
        append!(f.roots, Any[offsets64, bytes2])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == ["he", "llo"]

        offsets = Int32[0, 2, 3]
        bytes = UInt8[0x01, 0x02, 0xff]
        f = _cdata_fixture(
            "z",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets)), Ptr{Cvoid}(pointer(bytes))],
        )
        append!(f.roots, Any[offsets, bytes])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == [b"\x01\x02", b"\xff"]

        offsets64 = Int64[0, 1, 3]
        f = _cdata_fixture(
            "Z",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets64)), Ptr{Cvoid}(pointer(bytes))],
        )
        append!(f.roots, Any[offsets64, bytes])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == [b"\x01", b"\x02\xff"]

        offsets = Int32[123]
        f = _cdata_fixture("u", 0, Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets)), C_NULL])
        push!(f.roots, offsets)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == String[]

        offsets = Int32[123, 123, 123]
        f = _cdata_fixture("u", 2, Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets)), C_NULL])
        push!(f.roots, offsets)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == ["", ""]
    end

    @testset "fixed size binary" begin
        data = UInt8[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
        f = _cdata_fixture(
            "w:3",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(data))],
        )
        push!(f.roots, data)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == [(0x01, 0x02, 0x03), (0x04, 0x05, 0x06)]
    end

    @testset "list of primitives" begin
        child = _primitive_fixture("i", Int32[1, 2, 3, 4, 5])
        offsets = Int32[0, 2, 5]
        f = _cdata_fixture(
            "+l",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets))];
            children=[child],
        )
        push!(f.roots, offsets)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test collect(x) == [Int32[1, 2], Int32[3, 4, 5]]
    end

    @testset "fixed size list" begin
        child = _primitive_fixture("f", Float32[1, 2, 3, 4, 5, 6])
        f = _cdata_fixture("+w:3", 2, Ptr{Cvoid}[C_NULL]; children=[child])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f); convert=false)
        @test collect(x) == [(1.0f0, 2.0f0, 3.0f0), (4.0f0, 5.0f0, 6.0f0)]
    end

    @testset "struct root table with names and metadata" begin
        xchild = _primitive_fixture("i", Int32[1, 2, 3]; name="x", metadata=Dict("unit" => "id"))
        yoffsets = Int32[0, 1, 2, 3]
        ybytes = Vector{UInt8}(codeunits("abc"))
        ychild = _cdata_fixture(
            "u",
            3,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(yoffsets)), Ptr{Cvoid}(pointer(ybytes))];
            name="y",
        )
        append!(ychild.roots, Any[yoffsets, ybytes])
        root = _cdata_fixture(
            "+s",
            3,
            Ptr{Cvoid}[C_NULL];
            children=[xchild, ychild],
            metadata=Dict("source" => "cdata"),
        )
        tbl = Arrow.from_c_data(_schema_ptr(root), _array_ptr(root))
        @test Tables.columnnames(tbl) == [:x, :y]
        @test Tables.schema(tbl).types == (Int32, String)
        @test length(tbl) == 2
        @test Tables.rowcount(tbl) == 3
        @test collect(Tables.getcolumn(tbl, :x)) == Int32[1, 2, 3]
        @test collect(tbl.y) == ["a", "b", "c"]
        @test DataAPI.metadata(tbl, "source") == "cdata"
        @test DataAPI.colmetadata(tbl, :x, "unit") == "id"
        @test DataAPI.colmetadata(tbl) == Dict(:x => Dict("unit" => "id"))

        shortmeta_child = _primitive_fixture("i", Int32[1]; name="x", metadata=Dict("k" => "v"))
        shortmeta_root = _cdata_fixture(
            "+s",
            1,
            Ptr{Cvoid}[C_NULL];
            children=[shortmeta_child],
            metadata=Dict("x" => "yz"),
        )
        shortmeta_tbl = Arrow.from_c_data(_schema_ptr(shortmeta_root), _array_ptr(shortmeta_root))
        @test DataAPI.metadata(shortmeta_tbl, "x") == "yz"
        @test DataAPI.colmetadata(shortmeta_tbl, :x, "k") == "v"

        col = Tables.getcolumn(tbl, :x)
        GC.gc(true)
        @test collect(col) == Int32[1, 2, 3]
    end

    @testset "struct table offsets and owner roots" begin
        child = _primitive_fixture("i", Int32[10, 20, 30]; name="x")
        root = _cdata_fixture("+s", 2, Ptr{Cvoid}[C_NULL]; offset=Int64(1), children=[child])
        tbl = Arrow.from_c_data(_schema_ptr(root), _array_ptr(root))
        @test collect(Tables.getcolumn(tbl, :x)) == Int32[20, 30]

        col = let
            tbl2 = Arrow.from_c_data(_schema_ptr(root), _array_ptr(root))
            Tables.getcolumn(tbl2, :x)
        end
        GC.gc(true)
        GC.gc(true)
        @test collect(col) == Int32[20, 30]
        Arrow.release_c_data(col)

        child = _primitive_fixture("i", Int32[10, 20, 30]; offset=Int64(1), len=2, name="x")
        root = _cdata_fixture("+s", 2, Ptr{Cvoid}[C_NULL]; offset=Int64(1), children=[child])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(root), _array_ptr(root))
    end

    @testset "temporal and decimal formats" begin
        dates = Arrow.DATE[Arrow.DATE(1), Arrow.DATE(2)]
        f = _primitive_fixture("tdD", dates)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f); convert=false)
        @test collect(x) == dates
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f); convert=true)
        @test collect(x) == convert.(Dates.Date, dates)

        D = Arrow.Decimal{10,2,Int128}
        decimals = D[D(123), D(-45)]
        f = _primitive_fixture("d:10,2", decimals)
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f); convert=false)
        @test collect(x) == decimals
    end

    @testset "release behavior" begin
        f = _primitive_fixture("i", Int32[1, 2, 3])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test x[1] == 1
        Arrow.release_c_data(x)
        @test f.array[].release == C_NULL
        @test f.schema[].release == C_NULL
        @test_nowarn Arrow.release_c_data(x)
        @test_throws ArgumentError x[1]

        f = _primitive_fixture("i", Int32[1, 2, 3])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        owner = Arrow._owner(x)
        owner_lock = getfield(owner, :lock)
        started = Channel{Nothing}(1)
        locked = false
        lock(owner_lock)
        locked = true
        try
            task = Threads.@spawn begin
                put!(started, nothing)
                Arrow.release_c_data(x)
            end
            take!(started)
            sleep(0.05)
            @test f.array[].release != C_NULL
            unlock(owner_lock)
            locked = false
            wait(task)
        finally
            locked && unlock(owner_lock)
        end
        @test f.array[].release == C_NULL
        @test f.schema[].release == C_NULL
    end

    @testset "table release behavior" begin
        child = _primitive_fixture("i", Int32[1, 2, 3]; name="x")
        root = _cdata_fixture("+s", 3, Ptr{Cvoid}[C_NULL]; children=[child])
        tbl = Arrow.from_c_data(_schema_ptr(root), _array_ptr(root))
        col = Tables.getcolumn(tbl, :x)
        @test col[1] == 1
        Arrow.release_c_data(tbl)
        @test_nowarn Arrow.release_c_data(tbl)
        @test_throws ArgumentError Tables.getcolumn(tbl, :x)
        @test_throws ArgumentError col[1]
    end

    @testset "copy and collect own the result" begin
        f = _primitive_fixture("i", Int32[1, 2, 3])
        x = Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        a = collect(x)
        b = copy(x)
        Arrow.release_c_data(x)
        @test a == Int32[1, 2, 3]
        @test b == Int32[1, 2, 3]
    end

    @testset "malformed inputs" begin
        f = _primitive_fixture("i", Int32[1])
        @test_throws ArgumentError Arrow.from_c_data(
            Ptr{Arrow.ArrowSchema}(C_NULL),
            _array_ptr(f),
        )
        @test_throws ArgumentError Arrow.from_c_data(
            _schema_ptr(f),
            Ptr{Arrow.ArrowArray}(C_NULL),
        )

        f = _cdata_fixture("?", 0, Ptr{Cvoid}[])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))
        @test f.array[].release == C_NULL
        @test f.schema[].release == C_NULL

        f = _primitive_fixture("i", Int32[1])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            f.schema[].n_children,
            f.schema[].children,
            f.schema[].dictionary,
            C_NULL,
            f.schema[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            f.array[].n_buffers,
            f.array[].n_children,
            f.array[].buffers,
            f.array[].children,
            f.array[].dictionary,
            C_NULL,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.array[] = Arrow.ArrowArray(
            -1,
            0,
            0,
            2,
            0,
            f.array[].buffers,
            f.array[].children,
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            -1,
            f.array[].n_buffers,
            f.array[].n_children,
            f.array[].buffers,
            f.array[].children,
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            -1,
            f.array[].n_children,
            f.array[].buffers,
            f.array[].children,
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            f.array[].n_buffers,
            -1,
            f.array[].buffers,
            f.array[].children,
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            -1,
            f.schema[].children,
            f.schema[].dictionary,
            f.schema[].release,
            f.schema[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1]; null_count=Int64(2))
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1]; null_count=Int64(-1))
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _cdata_fixture("i", 1, Ptr{Cvoid}[C_NULL, C_NULL])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        child = _primitive_fixture("i", Int32[1])
        f = _cdata_fixture("+l", 1, Ptr{Cvoid}[C_NULL, C_NULL]; children=[child])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        offsets = Int32[0, 3, 2]
        bytes = UInt8[1, 2, 3]
        f = _cdata_fixture(
            "z",
            2,
            Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets)), Ptr{Cvoid}(pointer(bytes))],
        )
        append!(f.roots, Any[offsets, bytes])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        badmeta = reinterpret(UInt8, Int32[1, -1])
        f = _primitive_fixture("i", Int32[1])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            Cstring(pointer(badmeta)),
            f.schema[].flags,
            f.schema[].n_children,
            f.schema[].children,
            f.schema[].dictionary,
            f.schema[].release,
            f.schema[].private_data,
        )
        push!(f.roots, badmeta)
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            f.schema[].n_children,
            Ptr{Arrow.ArrowSchema}(UInt(0x01)),
            f.schema[].dictionary,
            f.schema[].release,
            f.schema[].private_data,
        )
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            f.array[].n_buffers,
            1,
            f.array[].buffers,
            Ptr{Ptr{Arrow.ArrowArray}}(C_NULL),
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        child = _primitive_fixture("i", Int32[1])
        schema_children = Ptr{Arrow.ArrowSchema}[_schema_ptr(child)]
        array_children = Ptr{Arrow.ArrowArray}[Ptr{Arrow.ArrowArray}(C_NULL)]
        f = _cdata_fixture("+s", 1, Ptr{Cvoid}[C_NULL])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            1,
            Ptr{Ptr{Arrow.ArrowSchema}}(pointer(schema_children)),
            f.schema[].dictionary,
            f.schema[].release,
            f.schema[].private_data,
        )
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            f.array[].n_buffers,
            1,
            f.array[].buffers,
            Ptr{Ptr{Arrow.ArrowArray}}(pointer(array_children)),
            f.array[].dictionary,
            f.array[].release,
            f.array[].private_data,
        )
        append!(f.roots, Any[child, schema_children, array_children])
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        dict_schema = Ref(f.schema[])
        f.schema[] = Arrow.ArrowSchema(
            f.schema[].format,
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            f.schema[].n_children,
            f.schema[].children,
            Base.unsafe_convert(Ptr{Arrow.ArrowSchema}, dict_schema),
            f.schema[].release,
            f.schema[].private_data,
        )
        push!(f.roots, dict_schema)
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        dict_array = Ref(f.array[])
        f.array[] = Arrow.ArrowArray(
            f.array[].length,
            f.array[].null_count,
            f.array[].offset,
            f.array[].n_buffers,
            f.array[].n_children,
            f.array[].buffers,
            f.array[].children,
            Base.unsafe_convert(Ptr{Arrow.ArrowArray}, dict_array),
            f.array[].release,
            f.array[].private_data,
        )
        push!(f.roots, dict_array)
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _primitive_fixture("i", Int32[1])
        fmt = fill(UInt8('i'), Arrow._CDATA_MAX_FORMAT_BYTES + 1)
        f.schema[] = Arrow.ArrowSchema(
            Cstring(pointer(fmt)),
            f.schema[].name,
            f.schema[].metadata,
            f.schema[].flags,
            f.schema[].n_children,
            f.schema[].children,
            f.schema[].dictionary,
            f.schema[].release,
            f.schema[].private_data,
        )
        push!(f.roots, fmt)
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        f = _cdata_fixture(
            "w:$(Arrow._CDATA_MAX_FIXED_SIZE + 1)",
            0,
            Ptr{Cvoid}[C_NULL, C_NULL],
        )
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(f), _array_ptr(f))

        child = _primitive_fixture("i", Int32[])
        for _ = 1:Arrow._CDATA_MAX_DEPTH
            offsets = Int32[0]
            child = _cdata_fixture(
                "+l",
                0,
                Ptr{Cvoid}[C_NULL, Ptr{Cvoid}(pointer(offsets))];
                children=[child],
            )
            push!(child.roots, offsets)
        end
        @test_throws ArgumentError Arrow.from_c_data(_schema_ptr(child), _array_ptr(child))
    end
end
