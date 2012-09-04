#include "global.h"

//////////////////////////////////////////////////////////////////////////////
///
/// @brief CUDA kernel to compute the values of an equally spaced capital
/// grid.
///
/// @details This CUDA kernel computes an equally spaced capital grid. The
/// upper and lower bounds are the deterministic steady-state values of
/// capital at the highest and lowest values of the TFP process
/// (respectively), scaled by 0.95 and 1.05 (respectively).
///
/// @param nk length of the capital grid.
/// @param nz length of the TFP grid.
/// @param beta discount factor.	       
/// @param alpha capital share of production.
/// @param Z pointer to grid of TFP values.
/// @param K pointer to grid of capital values.
///
/// @returns Void.
///
/// @author Eric M. Aldrich \n
///         ealdrich@ucsc.edu
///
/// @version 1.0
///
/// @date 24 July 2012
///
/// @copyright Copyright Eric M. Aldrich 2012 \n
///            Distributed under the Boost Software License, Version 1.0
///            (See accompanying file LICENSE_1_0.txt or copy at \n
///            http://www.boost.org/LICENSE_1_0.txt)
///
//////////////////////////////////////////////////////////////////////////////
__global__ void kGrid(const int nk, const int nz, const REAL beta,
		      const REAL alpha, const REAL delta, const REAL* Z,
		      REAL* K) 
{
  // thread
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  // grid for capital
  const REAL kmin = 0.95*pow((1/(alpha*Z[0]))*((1/beta)-1+delta),1/(alpha-1));
  const REAL kmax = 1.05*pow((1/(alpha*Z[nz-1]))*((1/beta)-1+delta),1/(alpha-1));
  const REAL kstep = (kmax-kmin)/(nk-1);
  K[i] = kmin + kstep*i;
}