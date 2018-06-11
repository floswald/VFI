
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
function poc_impl1()
	da = 1000
	dy = 5
	dp = 10
	dm = 30
	dt = 30
	dh = 3

	agrid = collect(range(0.1,step=0.001,length=da))
	Vplus = agrid .- 0.4*agrid.^2

	V = zeros(da*dy*dp*dm*dh*dt);

	# loop over all states
	for ia in eachindex(V,1)
		for iy in eachindex(V,1)
			for ip in eachindex(V,1)
				for ixm in eachindex(V,1)
					for it in eachindex(V,1)
						for ih in eachindex(V,1)
							V[sub2ind(ia,iy,ip,ixm,it,ih)] = vhm(ia,iy,ip,ixm,it,ih) + maximum(Vplus)
						end
					end
				end
			end
		end
	end
	return V
end

function poc1()
	@elapsed poc_impl1
end