#ifndef EMD_CUH_
#define EMD_CUH_

#include "cuda_helper.h"

template<typename T>
__global__ void approxmatch(const int b, const int n, const int m, const T * __restrict__ xyz1, const T * __restrict__ xyz2, T * __restrict__ match, T * temp){
	T * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	T multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ T buf[Block*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			float level=-powf(4.0f,j);
			if (j==-2){
				level=0;
			}
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				T x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				T suml=T(1e-9f);
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						T x2=xyz2[i*m*3+l0*3+l*3+0];
						T y2=xyz2[i*m*3+l0*3+l*3+1];
						T z2=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+0]=x2;
						buf[l*4+1]=y2;
						buf[l*4+2]=z2;
						buf[l*4+3]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						T x2=buf[l*4+0];
						T y2=buf[l*4+1];
						T z2=buf[l*4+2];
						T d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						T w=__expf(d)*buf[l*4+3];
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n)
					ratioL[k]=remainL[k]/suml;
			}
			/*for (int k=threadIdx.x;k<n;k+=gridDim.x){
				T x1=xyz1[i*n*3+k*3+0];
				T y1=xyz1[i*n*3+k*3+1];
				T z1=xyz1[i*n*3+k*3+2];
				T suml=1e-9f;
				for (int l=0;l<m;l++){
					T x2=xyz2[i*m*3+l*3+0];
					T y2=xyz2[i*m*3+l*3+1];
					T z2=xyz2[i*m*3+l*3+2];
					T w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*remainR[l];
					suml+=w;
				}
				ratioL[k]=remainL[k]/suml;
			}*/
			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				T x2=0,y2=0,z2=0;
				if (l<m){
					x2=xyz2[i*m*3+l*3+0];
					y2=xyz2[i*m*3+l*3+1];
					z2=xyz2[i*m*3+l*3+2];
				}
				T sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
						buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
						buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
						buf[k*4+3]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						T x1=buf[k*4+0];
						T y1=buf[k*4+1];
						T z1=buf[k*4+2];
						T w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*buf[k*4+3];
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=remainR[l];
					T consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
					ratioR[l]=consumption*remainR[l];
					remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
				}
			}
			/*for (int l=threadIdx.x;l<m;l+=blockDim.x){
				T x2=xyz2[i*m*3+l*3+0];
				T y2=xyz2[i*m*3+l*3+1];
				T z2=xyz2[i*m*3+l*3+2];
				T sumr=0;
				for (int k=0;k<n;k++){
					T x1=xyz1[i*n*3+k*3+0];
					T y1=xyz1[i*n*3+k*3+1];
					T z1=xyz1[i*n*3+k*3+2];
					T w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k];
					sumr+=w;
				}
				sumr*=remainR[l];
				T consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
				ratioR[l]=consumption*remainR[l];
				remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
			}*/
			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				T x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				T suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
						buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
						buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+3]=ratioR[l0+l];
					}
					__syncthreads();
					T rl=ratioL[k];
					if (k<n){
						for (int l=0;l<lend;l++){
							T x2=buf[l*4+0];
							T y2=buf[l*4+1];
							T z2=buf[l*4+2];
							T w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*rl*buf[l*4+3];
							match[i*n*m+(l0+l)*n+k]+=w;
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			/*for (int k=threadIdx.x;k<n;k+=blockDim.x){
				T x1=xyz1[i*n*3+k*3+0];
				T y1=xyz1[i*n*3+k*3+1];
				T z1=xyz1[i*n*3+k*3+2];
				T suml=0;
				for (int l=0;l<m;l++){
					T x2=xyz2[i*m*3+l*3+0];
					T y2=xyz2[i*m*3+l*3+1];
					T z2=xyz2[i*m*3+l*3+2];
					T w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k]*ratioR[l];
					match[i*n*m+l*n+k]+=w;
					suml+=w;
				}
				remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}*/
			__syncthreads();
		}
	}
}

void approx_match(const int b, const int n, const int m, const at::Tensor xyz1, const at::Tensor xyz2, at::Tensor match, at::Tensor temp){
	AT_DISPATCH_FLOATING_TYPES(match.type(), "approxmatch", ([&] {
		approxmatch
			<<<32, 512>>>(
			b, n, m, 
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			temp.data<scalar_t>());
	}));
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError())
}

template<typename T>
__global__ void matchcost(const int b, const int n, const int m, const T * __restrict__ xyz1, const T * __restrict__ xyz2, const T * __restrict__ match, T * __restrict__ out){
	__shared__ T allsum[512];
	const int Block=1024;
	__shared__ T buf[Block*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		T subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			T x1=0,y1=0,z1=0;
			if (k<n){
				x1=xyz1[i*n*3+k*3+0];
				y1=xyz1[i*n*3+k*3+1];
				z1=xyz1[i*n*3+k*3+2];
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
					buf[l]=xyz2[i*m*3+l0*3+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						T x2=buf[l*3+0];
						T y2=buf[l*3+1];
						T z2=buf[l*3+2];
						T d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						subsum+=d*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=allsum[0];
		__syncthreads();
	}
}

void match_cost(const int b, const int n, const int m, const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match, at::Tensor out){
	AT_DISPATCH_FLOATING_TYPES(xyz1.type(), "matchcost", ([&] {
		matchcost<<<32, 512>>>(
			b, n, m,
			xyz1.data<scalar_t>(),
			xyz2.data<scalar_t>(),
			match.data<scalar_t>(),
			out.data<scalar_t>());
	}));
	CUDA_CHECK(cudaGetLastError())
}

#endif