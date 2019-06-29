#include <ATen/ATen.h>

#include <vector>

#include "cuda/emd.cuh"


std::vector<at::Tensor> emd_forward_cuda(
	at::Tensor xyz1, // B x N1 x D
	at::Tensor xyz2) // B x N2 x D
{
	// Some useful values
	const int64_t batch_size = xyz1.size(0);
	const int64_t num_pts_1 = xyz1.size(1);
	const int64_t num_pts_2 = xyz2.size(1);
	// const int64_t dim = xyz1.size(2);

	// Allocate necessary data structures
	at::Tensor match = at::zeros({batch_size, num_pts_1, num_pts_2}, 
		xyz1.options());
	at::Tensor cost = at::zeros({batch_size}, xyz1.options());
	at::Tensor temp = at::zeros({batch_size, 2 * (num_pts_1 + num_pts_2)}, 
		xyz1.options());

	float xyz1_f = xyz1.data<float>();
	float xyz2_f = xyz2.data<float>();
	float match_f = match.data<float>();
	float temp_f = temp.data<float>();
	float cost_f = cost.data<float>();

	// Find the approximate matching
	approxmatchLauncher(
		batch_size, num_pts_1, num_pts_2,
		&xyz1_f,
		&xyz2_f, 
		&match_f,
		&temp_f
	);

	// Compute the matching cost
	matchcostLauncher(
		batch_size, num_pts_1, num_pts_2, 
		&xyz1_f,
		&xyz2_f, 
		&match_f,
		&cost_f
	);

	return {cost, match};
}
