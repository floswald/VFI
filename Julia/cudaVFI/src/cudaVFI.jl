module cudaVFI

	using JSON

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
			this.kgrid        = range(kmin,step                                                               = (kmax-kmin)/(p.nk-1),length = p.nk)
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
