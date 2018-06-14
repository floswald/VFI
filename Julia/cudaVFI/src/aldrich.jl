greet() = print("Hello World!")

mutable struct Param
	eta :: Float64
	beta  :: Float64
	alpha :: Float64
	delta :: Float64
	mu    :: Float64
	rho   :: Float64
	sigma   :: Float64
	nk    :: Int
	nz    :: Int
	tol   :: Float64
	function Param(;par=Dict())
        f=open(joinpath(dirname(@__FILE__),"..","..","..","params.json")) 
		j = JSON.parse(f)
		close(f)
    	this = new()
    	for (k,v) in j
            setfield!(this,Symbol(k),v["value"])
    	end
        if length(par) > 0
            # override parameters from dict p
            for (k,v) in par
                setfield!(this,k,v)
            end
        end
    	return this
	end

end

mutable struct Model 
	V       :: Matrix{Float32}   # value fun
	V0      :: Matrix{Float32}   # value fun
	G       :: Matrix{Int}   # policy fun
	G0      :: Matrix{Int}   # policy fun
	P       :: Matrix{Float32}   # transition matrix
	zgrid   :: Vector{Float32}
	kgrid   :: StepRangeLen{Float32}
	fkgrid  :: Vector{Float32}
	ydepK   :: Matrix{Float32}
	counter :: Int
	function Model(p::Param)
		this              = new()
		this.V            = zeros(Float32,p.nk,p.nz)
		this.G            = zeros(Int,p.nk,p.nz)
		this.V0           = zeros(Float32,p.nk,p.nz)
		this.G0           = zeros(Int,p.nk,p.nz)
		this.zgrid,this.P = rouwenhorst(p.rho,p.mu,p.sigma,p.nz)
		this.zgrid = exp.(this.zgrid)
		kmin              = 0.95*(((1/(p.alpha*this.zgrid[1]))*((1/p.beta)-1+p.delta))^(1/(p.alpha-1)))
		kmax              = 1.05*(((1/(p.alpha*this.zgrid[end]))*((1/p.beta)-1+p.delta))^(1/(p.alpha-1)))
		this.kgrid        = range(kmin,step = (kmax-kmin)/(p.nk-1),length = p.nk)
		this.fkgrid       = (this.kgrid).^p.alpha
		this.counter      = 0
		# output plus depreciated capital
		this.ydepK = this.fkgrid .* this.zgrid' .+ (1-p.delta).*repeat(this.kgrid,1,p.nz)
		return this
	end
end

mutable struct CuModel 
	V       :: CuMatrix{Float32}   # value fun
	V0      :: CuMatrix{Float32}   # value fun
	G       :: CuMatrix{Int}   # policy fun
	G0      :: CuMatrix{Int}   # policy fun
	P       :: CuMatrix{Float32}   # transition matrix
	zgrid   :: CuVector{Float32}
	kgrid   :: CuVector{Float32}
	fkgrid  :: CuVector{Float32}
	ydepK   :: CuMatrix{Float32}
	counter :: Int
	function CuModel(m::Model)
		this         = new()
		this.V       = CuArray(m.V)
		this.G       = CuArray(m.G)
		this.V0      = CuArray(m.V0)
		this.G0      = CuArray(m.G0)
		this.P       = CuArray(m.P)
		this.zgrid   = CuArray(m.zgrid)
		this.kgrid   = CuArray(convert(Vector{Float32},collect(m.kgrid)))
		this.counter = 0
		# output plus depreciated capital
		this.ydepK = CuArray(m.ydepK)
		return this
	end
end

ufun(x::StepRangeLen{Float32},p::Param) = (x.^(1-p.eta))/(1-p.eta)
ufun(x::Float32,eta::Float32) = (x^(1-eta))/(1-eta)


function rouwenhorst(rho::Float64,mu_eps::Float64,sigma_eps::Float64,n::Int)
	q = (rho+1)/2
	nu = ((n-1)/(1-rho^2))^(1/2) * sigma_eps
	P = reshape([q,1-q,1-q,q],2,2)

	for i=2:n-1

		P = q * vcat(hcat(P , zeros(i,1)),zeros(1,i+1)) .+ (1-q).* vcat( hcat(zeros(i,1),P), zeros(1,i+1)) .+ 
		(1-q) .* vcat(zeros(1,i+1),hcat(P,zeros(i,1))) .+ q .*vcat(zeros(1,i+1),hcat(zeros(i,1),P))
		P[2:i,:] = P[2:i,:] ./ 2

	end
	z = collect(range(mu_eps/(1-rho)-nu,step=2*nu/(n-1),length=n));
	return (z,P)
end

function update(m::Model,p::Param)

	for i in 1:p.nk
		for j in 1:p.nz
			# constraints on future capital grid
			klo = 1
			khi = searchsortedlast(m.kgrid,m.ydepK[i,j])
			khi = khi > 0 ? khi-1 : khi 

			# number of feasible points
			# nksub = khi-klo+1

			# compute EV at all poitns (not only the nksub ones)
			Exp = view(m.V0,klo:khi,:)*m.P[j,:]

			w = ufun(m.ydepK[i,j] .- m.kgrid[klo:khi],p) .+ p.beta*Exp
			v,g = findmax(w)
			m.V[i,j] = v
			m.G[i,j] = g + (klo-1)
		end
	end
	differ = maximum(abs,m.V.-m.V0)
	m.V0[:,:] = m.V
	m.counter += 1
	return differ
end

function model() 
	p=Param();
	m=Model(p);
	(m,p)
end

function gpu_launcher(m::Model,p::Param)
	V       = CuArray(m.V)
	G       = CuArray(m.G)
	V0      = CuArray(m.V0)
	G0      = CuArray(m.G0)
	P       = CuArray(m.P)
	zgrid   = CuArray(m.zgrid)
	kgrid   = CuArray(convert(Vector{Float32},collect(m.kgrid)))
	counter = 0
	ydepK = CuArray(m.ydepK)
	n = length(V)

	ma = CuVector{Float32}(1)
	w0 = Array{Float32,1}(undef,length(kgrid))
	fill!(w0,typemin(Float32))

	ix = CuVector{Int}(1)

	# blocking setup
	ctx = CuCurrentContext()
    dev = device(ctx)

 #    total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
 #    threads_x = floor(Int, total_threads / size(V,2))
 #    threads_y = total_threads ÷ threads_x
 #    threads = (threads_x, threads_y)
	# blocks = ceil.(Int, n ./ threads)
	cols = size(V,2)
	rows = size(V,1)
	# @info("launch GPU on $cols blocks, and $rows threads")

	differ = 10.0

	while abs(differ) > p.tol
		w = CuArray(w0)
		# @cuda blocks=blocks threads=threads update_kernel(V,V0,G,ydepK,kgrid,m,ix,P,p.beta,p.eta)
		@cuda blocks=cols threads=rows cudaVFI.update_kernel(V,V0,G,ydepK,kgrid,ma,w,ix,P,p.beta,p.eta)

		# sync_threads()
		v1 = Array(V)
		v0 = Array(V0)
		# if m.counter==5
			# println(v1.-v0)
		# 	error()
		# end
		differ = maximum(abs,v1.-v0)
		# if mod(m.counter,50)==0
			@info("count: $(m.counter), diff=$differ")
			# println("v1 = $v1")
			# println("v0 = $v0")
		# end
		# m.V0[:,:] = m.V  # update iteration array
		copyto!(V0,v1)
		# V0 = CuArray(v0)
		m.counter += 1
	end
	copyto!(m.V,Array(V))
	return m
end	


function update_kernel(V::CuDeviceMatrix{Float32},
	                   V0::CuDeviceMatrix{Float32},
	                   G::CuDeviceMatrix{Int},
	                   ydepK::CuDeviceMatrix{Float32},
	                   kgrid::CuDeviceVector{Float32},
	                   m::CuDeviceVector{Float32},
	                   w::CuDeviceVector{Float32},
	                   ix::CuDeviceVector{Int},
	                   P::CuDeviceMatrix{Float32},
	                   beta::Float64,eta::Float64)

	# block x thread -> array index
	iz = blockIdx().x
    nz = gridDim().x

    ik = threadIdx().x
	nk = blockDim().x

	if iz <= nz && ik <= nk

		# ik = (blockIdx().x-1) * blockDim().x + threadIdx().x
		# iz = (blockIdx().y-1) * blockDim().y + threadIdx().y
		# iz = 1
		# @cuprintf("iz = %d\n",iz)
		# @cuprintf("ik = %d\n",ik)

		# bounds on choice space
		klo = 1
		# @cuprintf("[ik,iz] = %d\n",iz)
		# @cuprintf("[iz,ik] = [ %ld , %ld ]\n",iz,ik)
		# @cuprintf("iz = %d\n",iz)
		# @cuprintf("ydepK[1,1] = %f\n",ydepK[1,1])
		khi = searchsortedlast(kgrid,ydepK[ik,iz])
		khi = khi > 0 ? khi-1 : khi 
		@assert(khi>0)
		# if ik==1 & iz==1
		# 	# @cuprintf("comparing kgrid vs ydepK[ik,iz]: %lf vs %lf\n",Float64(kgrid[end]),Float64(ydepK[ik,iz]))
		# 	@cuprintf("klo = %ld, khi = %ld\n",klo,khi)
		# end

		# expected value
		Exp = 0.0
		# for iik in 1:khi 
		# 	for iiz in 1:size(V,2)
		# 		Exp += P[iz,iiz] * V0[iik,iiz]
		# 	end
		# end
		# x = Float32(ydepK[1,1])
		# w[1] = CUDAnative.pow(x,2.0)
		# maximization Vector
		for i in 1:khi
			# CUDAnative.pow(1.0,2.0)#/(1-eta) + beta * Exp 
			Exp = 0.0
			for iiz in 1:size(V,2)
				Exp += P[iz,iiz] * V0[i,iiz]
			end


			# CUDAnative.pow(ydepK[ik,iz] - kgrid[i],1-eta)#/(1-eta) + beta * Exp 
			# @cuprintf("ydepK[ik,iz] - kgrid[i]= %lf\n",convert(Float64,ydepK[ik,iz] - kgrid[i]))
		# if ik==1 & iz==1
		# @cuprintf("on block %ld, thread %ld, ik=%ld, iz=%ld. Exp= %lf\n", Int64(blockIdx().x), Int64(threadIdx().x), Int(ik),Int(iz),Exp)
		# end
			w[i] = CUDAnative.pow(convert(Float64,ydepK[ik,iz] - kgrid[i]),1-eta)/(1-eta) + beta * Exp 
			@cuprintf("on block %ld, thread %ld, ik=%ld, iz=%ld, ufun = %lf, w = %lf,khi=%ld\n",Int64(blockIdx().x), Int64(threadIdx().x), Int(ik),Int(iz),CUDAnative.pow(convert(Float64,ydepK[ik,iz] - kgrid[i]),1-eta)/(1-eta),Float64(w[i]),khi)
			# w[i] = Float64(i) + beta * Exp 
			# @cuprintf("on block %ld, thread %ld, w[%ld] = %lf\n", Int64(blockIdx().x), Int64(threadIdx().x), i, Float64(w[i]))
			# if ydepK[ik,iz] - kgrid[i] < 0
			# 	@cuprintf("ydepK[ik,iz] - kgrid[i]= %lf, w[i] = %lf\n",convert(Float64,ydepK[ik,iz] - kgrid[i]),Float64(w[i]))
			# end
			# @cuprintf("w[%ld] = %lf\n",i,convert(Float64,w[i]))
			# w[i] = CUDAnative.pow(1.0,1-eta)/(1-eta) + beta * Exp 
			# w[i] = CUDAnative.pow(1.0,2.0) + beta * Exp 
			# w[i] = CUDAnative.pow(x,2.0) + beta * Exp 
		end

		max_kernel_ub(w,m,ix,khi)

		# if ik==1 & iz==1
		# @cuprintf("on block %ld, thread %ld, ik=%ld, iz=%ld. Exp= %lf\n", Int64(blockIdx().x), Int64(threadIdx().x), Int(ik),Int(iz),Exp)
		# end
		# @cuprintf("on block %ld, thread %ld, ik=%ld, iz=%ld. max = %lf, id = %ld\n", Int64(blockIdx().x), Int64(threadIdx().x), Int(ik),Int(iz),Float64(m[1]),Int64(ix[1]))
	# end
		# else 
			# m[1] = 0
			# ix[1] = -1
		# end
		# @cuprintf("index: [%ld,%ld], max = %lf, id = %ld\n",ik,iz,Float64(m[1]),ix[1])

		V[ik,iz] = m[1]
		# V[ik,iz] = ik + iz
		# V[ik,iz] = Float32(1.0)
		G[ik,iz] = ix[1]
		return nothing
	end
end

function runGPU(;par=Dict())
	p = Param(par=par)
	m = Model(p)
	m = gpu_launcher(m,p)
	return m
end

function runCPU()
	# p = Param(Dict(:nk=>nk))
	p = Param()
	m = Model(p)
	differ = 10.0
	while abs(differ) > p.tol
		differ = update(m,p)
		if mod(m.counter,50)==0
			@info("count: $(m.counter), diff=$differ")
		end
	end
	return m
end

# pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
#                              rowresult::CuDeviceMatrix{Float32}, n)

# function kernel_vadd(a, b, c)
#     i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#     c[i] = a[i] + b[i]

#     return nothing
# end

function pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
                          rowresult::CuDeviceMatrix{Float32}, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # store to shared memory
        shmem = @cuDynamicSharedMem(Float32, 2*blockDim().x + 2*blockDim().y)
        if threadIdx().y == 1
            shmem[threadIdx().x] = lat[i]
            shmem[blockDim().x + threadIdx().x] = lon[i]
        end
        if threadIdx().x == 1
            shmem[2*blockDim().x + threadIdx().y] = lat[j]
            shmem[2*blockDim().x + blockDim().y + threadIdx().y] = lon[j]
        end
        sync_threads()

        # load from shared memory
        lat_i = shmem[threadIdx().x]
        lon_i = shmem[blockDim().x + threadIdx().x]
        lat_j = shmem[2*blockDim().x + threadIdx().y]
        lon_j = shmem[2*blockDim().x + blockDim().y + threadIdx().y]

        @inbounds rowresult[i, j] = my_gpu(lat_i, lon_i, lat_j, lon_j)
        # @inbounds rowresult[i, j] = haversine_gpu(lat_i, lon_i, lat_j, lon_j, 6372.8f0)
    end
end

function my_gpu(xi::Float32,yi::Float32,xj::Float32,yj::Float32)
	return 2*xi + 3*yi - xj*yj
end

function pairwise_dist_gpu(lat::Vector{Float32}, lon::Vector{Float32})
    # upload
    lat_gpu = CuArray(lat)
    lon_gpu = CuArray(lon)

    # allocate
    n = length(lat)
    rowresult_gpu = CuArray{Float32}(n, n)

    # calculate launch configuration
    # NOTE: we want our launch configuration to be as square as possible,
    #       because that minimizes shared memory usage
    ctx = CuCurrentContext()
    dev = device(ctx)
    total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
    threads_x = floor(Int, sqrt(total_threads))
    threads_y = total_threads ÷ threads_x
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, n ./ threads)

    # calculate size of dynamic shared memory
    shmem = 2 * sum(threads) * sizeof(Float32)

    @cuda blocks=blocks threads=threads shmem=shmem pairwise_dist_kernel(lat_gpu, lon_gpu, rowresult_gpu, n)
	return Array(rowresult_gpu)
end

# function runGPU()
# 	p = Param()
# 	m = Model(p)
# 	differ = 10.0
# 	while abs(differ) > p.tol
# 		differ = update(m,p)
# 		if mod(m.counter,10)==0
# 			@info("count: $(m.counter), diff=$differ")
# 		end
# 	end
# 	return m
# end

function shootout()

	@info("running both once to precompile")
	runCPU(2);
	runGPU(2);
	println()

	for nk in range(100,step=50,length=5)
		@info("now timing at nk=$nk:")
		cpu = Base.@elapsed mc=runCPU(nk)
		GC.gc()
		gpu = CUDAdrv.@elapsed mg=runGPU(nk)
		GC.gc()
		maxdiff = maximum(abs,mc.V .- mg.V)
		@info("cpu = $cpu")
		@info("gpu = $gpu")
		@info("cpu/gpu = $(cpu/gpu)")
		@info("maxdiff = $maxdiff")
		println()
	end
end




# function kernel_vadd(a, b, c)
#     i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#     @cuprintf("blockx= %d, blockDx=%d, threadid=%d\n",(blockIdx().x-1) , blockDim().x , threadIdx().x)
#     c[i] = a[i] + b[i]

#     return nothing
# end


# a = round.(rand(Float32, (3, 4)))
# b = round.(rand(Float32, (3, 4)))
# d_a = CuArray(a)
# d_b = CuArray(b)
# d_c = similar(d_a)  # output array

# # run the kernel and fetch results
# # syntax: @cuda [kwargs...] kernel(args...)
# @cuda threads=12 kernel_vadd(d_a, d_b, d_c)

# function kernel(ydepK::CuDeviceMatrix{Float32})
# 	x = ydepK[1,1]
# 	# x = 1.0
# 	y = CUDAnative.pow(convert(Float64,x),2.0)
# 	return nothing
# end

# function cutest()
# 	y = rand(Float32,3,4)
# 	cuy = CuArray(y)
# 	@cuda blocks=2 threads=2 kernel(cuy)
# end

