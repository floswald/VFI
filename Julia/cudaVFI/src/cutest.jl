
# /opt/cuda-8.0/samples/1_Utilities/deviceQuery/deviceQuery


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

function launcher2d(f::Function,nthread,nblocks)

	x = CuArray(zeros(Int,nthread,nblocks))
	y = CuArray(zeros(Int,nthread,nblocks))

    @cuda blocks=nblocks threads=nthread f(x,y)
    return x 
end

function kernelid2d(x::CuDeviceMatrix{Int},y::CuDeviceMatrix{Int})
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
	x[ix,iy] = threadIdx().x
	y[ix,iy] = threadIdx().y
end


function blockid2d(y::CuDeviceVector{Int})
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	y[idx] = blockIdx().x
end

vhm(ia::Int,iy::Int,ip::Int,ixm::Int,it::Int,ih::Int) = ( 1.1*ia + 0.5*iy + 3.3*ip + 0.1*ixm + 0.4*it ) * ih

# proof of concept
function poc_cpu1()
	da = 500
	dy = 5
	dp = 10
	dm = 30
	dt = 30
	dh = 3

	agrid = collect(range(0.1,step=0.001,length=da))
	Vplus = agrid .- 0.4*agrid.^2

	V = zeros(Float32,da,dy,dp,dm,dh,dt);
	iV = zeros(Int,da,dy,dp,dm,dh,dt);

	# loop over all states
	for ia in axes(V,1)
		for iy in axes(V,2)
			for ip in axes(V,3)
				for ixm in axes(V,4)
					for it in axes(V,5)
						for ih in axes(V,6)
							w = agrid .- 0.4*agrid.^2 .- vhm(ia,iy,ip,ixm,it,ih)
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

function poc_gpu1()
	da = 500
	dy = 5
	dp = 10
	dm = 30
	dt = 30
	dh = 3

	agrid = collect(range(0.1,step=0.001,length=da))
	a = CuArray(agrid)
	V = CuArray{Float32}(da,dy,dp,dm,dh,dt);
	iV = CuArray{Int}(da,dy,dp,dm,dh,dt);

	@cuda blocks=1000 threads=length(V)/1000 poc_kernel(V,iV,a)

end

function poc_kernel(V::CuDeviceArray{Float32},iV::CuDeviceArray{Int},a::CuDeviceVector{Float32})
	idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
	ii = CartesianIndices(V)[idx]
	w = zeros(a)
	for i in 1:length(w)
		w[i] = a - 0.4*a^2 - vhm(Tuple(ii)...)
	end
	v,i = findmax(w)
	V[idx] = v
	iV[idx] = i
end



function poc1()
	@elapsed poc_cpu1()
	CUDAdrv.@elapsed poc_gpu1()
end