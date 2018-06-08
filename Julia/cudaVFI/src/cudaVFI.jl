module cudaVFI

	using JSON
	using CUDAnative
	using CUDAdrv

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
	    	return this
		end

	end

	mutable struct Model 
		V       :: Matrix{Float64}   # value fun
		V0      :: Matrix{Float64}   # value fun
		G       :: Matrix{Int}   # policy fun
		G0      :: Matrix{Int}   # policy fun
		P       :: Matrix{Float64}   # transition matrix
		zgrid   :: Vector{Float64}
		kgrid   :: StepRangeLen{Float64}
		fkgrid  :: Vector{Float64}
		ydepK   :: Matrix{Float64}
		counter :: Int
		function Model(p::Param)
			this              = new()
			this.V            = zeros(Float64,p.nk,p.nz)
			this.G            = zeros(Int,p.nk,p.nz)
			this.V0           = zeros(Float64,p.nk,p.nz)
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


	ufun(x::StepRangeLen{Float64},p::Param) = (x.^(1-p.eta))/(1-p.eta)


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

	function runCPU()
		p = Param()
		m = Model(p)
		differ = 10.0
		while abs(differ) > p.tol
			differ = update(m,p)
			if mod(m.counter,10)==0
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

	function runGPU()
		p = Param()
		m = Model(p)
		differ = 10.0
		while abs(differ) > p.tol
			differ = update(m,p)
			if mod(m.counter,10)==0
				@info("count: $(m.counter), diff=$differ")
			end
		end
		return m
	end

end # module
