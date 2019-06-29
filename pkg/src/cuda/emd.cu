#include <ATen/ATen.h>

#include <vector>

#include "cuda/emd.cuh"


std::vector<at::Tensor> emd_forward_cuda(
	at::Tensor xyz1, // B x N1 x D
	at::Tensor xyz2) // B x N2 x D
{
	// Some useful values
	const int batch_size = xyz1.size(0);
	const int num_pts_1 = xyz1.size(1);
	const int num_pts_2 = xyz2.size(1);
	// const int dim = xyz1.size(2);

	// Allocate necessary data structures
	at::Tensor match = at::zeros({batch_size, num_pts_1, num_pts_2}, 
		xyz1.options());
	at::Tensor cost = at::zeros({batch_size}, xyz1.options());
	at::Tensor temp = at::zeros({batch_size, 2 * (num_pts_1 + num_pts_2)}, 
		xyz1.options());

	// Find the approximate matching
	approx_match(
		batch_size, num_pts_1, num_pts_2,
		xyz1,
		xyz2, 
		match,
		temp
	);

	// Compute the matching cost
	match_cost(
		batch_size, num_pts_1, num_pts_2, 
		xyz1,
		xyz2, 
		match,
		cost
	);

	return {cost, match};
}
