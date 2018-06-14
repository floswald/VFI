
# /usr/local/cuda-9.2/samples/1_Utilities/deviceQuery/deviceQuery


function launcher(f::Function,nthread,nblocks)

	x = CuArray(zeros(Int,nthread*nblocks))

    @cuda blocks=nblocks threads=nthread f(x)
    return x
end

function kernelid(y::CuDeviceVector{Int})
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	y[idx] = threadIdx().x
end


function blockid(y::CuDeviceVector{Int})
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	y[idx] = blockIdx().x
end

function launcher2d(f::Function,dims,nthread,nblocks)

	x = CuArray(zeros(Int,dims))
	y = CuArray(zeros(Int,dims))

    @cuda blocks=nblocks threads=nthread f(x,y)
    return (Array(x),Array(y))
end

function kernelid2d(x::CuDeviceMatrix{Int},y::CuDeviceMatrix{Int})
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
	x[ix,iy] = threadIdx().x
	y[ix,iy] = threadIdx().y
end


function blockid2d(x::CuDeviceMatrix{Int},y::CuDeviceMatrix{Int})
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
	x[ix,iy] = blockIdx().x
	y[ix,iy] = blockIdx().y
end

vhm(ia::Int,iy::Int,ip::Int,ixm::Int,it::Int,ih::Int) = ( 1.1*ia + 0.5*iy + 3.3*ip + 0.1*ixm + 0.4*it ) * ih

# proof of concept
function poc_cpu1(;na::Int=100)
	da = na
	dy = 5
	dp = 10
	dm = 30
	dt = 30
	dh = 3

	agrid = collect(range(0.1,step=0.001,length=da))
	Vplus = agrid .- 0.4*agrid.^2

	V = zeros(Float32,da,dy,dp,dm,dh,dt);
	iV = zeros(Int,da,dy,dp,dm,dh,dt);

	@info("number of elements in V: $(length(V))")

	# loop over all states
	for ia in axes(V,1)
		for iy in axes(V,2)
			for ip in axes(V,3)
				for ixm in axes(V,4)
					for it in axes(V,5)
						for ih in axes(V,6)
							w = agrid .- 0.4*agrid.^2 #.+ LinearIndices(V)[ia,iy,ip,ixm,it,ih]
							v,i = findmax(w)
							V[LinearIndices(V)[ia,iy,ip,ixm,it,ih]] = v
							iV[LinearIndices(V)[ia,iy,ip,ixm,it,ih]] = i
						end
					end
				end
			end
		end
	end
	return V
end

function poc_gpu1(;na::Int=100)
	da = na
	dy = 5
	dp = 10
	dm = 30
	dt = 30
	dh = 3

	agrid = convert(Array{Float32},collect(range(0.1,step=0.001,length=da)))
	a = CuArray(agrid)
	w = CuArray(agrid)
	V = CuArray{Float32}(da,dy,dp,dm,dh,dt);
	iV = CuArray{Int}(da,dy,dp,dm,dh,dt);

	m = CuArray{Float32}(1)
	ix = CuArray{Int}(1)

	n = length(V)

	ctx = CuCurrentContext()
    dev = device(ctx)
	total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
	blocks = ceil(Int, n / total_threads)



	@cuda blocks=blocks threads=total_threads poc_kernel(V,iV,a,w,m,ix)
	return Array(V)

end
function max_kernel_ub(v::CuDeviceVector{Float32},m::CuDeviceVector{Float32},ix::CuDeviceVector{Int},khi::Int)
	r = typemin(Float32)
	id = 0
	for i in 1:khi
		# @cuprintf("v[i] = %lf\n",Float64(v[i]))
		# @cuprintf("r = %lf,ix = %ld\n",r,ix[1])
		if v[i] > r
			r = v[i]
			id = i
		end
	end
	m[1] = r
	ix[1] = id
		# @cuprintf("returning m[1]= %lf,ix = %ld\n",Float64(m[1]),Int64(ix[1]))
	return nothing
	#return r
end



function max_kernel(v::CuDeviceVector{Float32},m::CuDeviceVector{Float32},ix::CuDeviceVector{Int})
	r = typemin(Float32)
	id = 0
	for i in 1:length(v)
		# @cuprintf("v[i] = %lf\n",Float64(v[i]))
		# @cuprintf("r = %lf,ix = %ld\n",r,ix[1])
		if v[i] > r
			r = v[i]
			id = i
		end
	end
	m[1] = r
	ix[1] = id
		# @cuprintf("returning m[1]= %lf,ix = %ld\n",Float64(m[1]),Int64(ix[1]))
	return nothing
	#return r
end

function poc_kernel(V::CuDeviceArray{Float32},iV::CuDeviceArray{Int},
	a::CuDeviceVector{Float32},w::CuDeviceVector{Float32},m::CuDeviceVector{Float32},ix::CuDeviceVector{Int})
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	# ii = CartesianIndices(V)[idx]
	for i in 1:length(w)
		# w[i] = a - 0.4*a^2 - vhm(Tuple(ii)...)
		w[i] = a[i] - 0.4*a[i]^2
	end
	v = max_kernel(w,m,ix)
	V[idx] = m[1]
	iV[idx] = ix[1]
	return nothing
end

function state_kernel3D(V::CuDeviceArray{Float32},iV::CuDeviceArray{Float32})

	n = size(V)
	@assert length(n)==3
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	# idx = threadIdx().x
	# what linear index of the array V is that ???
	ix = CUDAnative.mod(Float64(n[1]),Float64(idx))
	# ix = mod(idx,n[1])
	# iy = mod(, n[2])
	# iy = mod(idx / n[1] ,n[2])
	iy = CUDAnative.mod(Float64(n[2]),Float64(idx / n[1] ))
	# iz = idx / (n[1]*n[2])
	iz = idx / (n[1]*n[2])
	q = ix + n[1]*(iy-1) + n[1]*n[2]*(iz-1)
	@cuprintf("q = %lf\n",q)
	iV[idx] = q
	return nothing
end

function test3D()
	V = rand(Float32,3,4,5)
	iV = zeros(Float32,size(V))
	d_V = CuArray(V)
	d_iV = CuArray(iV)
	n = length(V)

	@cuda threads=n state_kernel3D(d_V,d_iV)

	return Array(d_iV)
end




function poc1()
	@info("running both once to precompile")
	poc_cpu1(na=2);
	poc_gpu1(na=2);
	println()


	for na in range(100,step=50,length=5)
		@info("now timing at na=$na:")
		cpu = Base.@elapsed vcpu=poc_cpu1(na=na)
		GC.gc()
		gpu = CUDAdrv.@elapsed vgpu=poc_gpu1(na=na)
		GC.gc()
		@info("cpu = $cpu")
		@info("gpu = $gpu")
		@info("cpu/gpu = $(cpu/gpu)")
		@info("maxabs(diff) = $(maximum(abs,vcpu.-vgpu))")
		println()
	end

end