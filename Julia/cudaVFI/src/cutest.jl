
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