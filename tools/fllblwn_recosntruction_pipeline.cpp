/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include <sstream>

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>


using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

int default_polynomial_order = 2;
bool default_use_polynomial_fit = false;
double default_search_radius = 0.0,
    default_sqr_gauss_param = 0.0;


void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s input.pcd output.pcd <options>\n", argv[0]);
  print_info ("  where options are:\n");
  print_info ("                     -radius X          = sphere radius to be used for finding the k-nearest neighbors used for fitting (default: ");
  print_value ("%f", default_search_radius); print_info (")\n");
  print_info ("                     -sqr_gauss_param X = parameter used for the distance based weighting of neighbors (recommended = search_radius^2) (default: ");
  print_value ("%f", default_sqr_gauss_param); print_info (")\n");
  print_info ("                     -use_polynomial_fit X = decides whether the surface and normal are approximated using a polynomial or only via tangent estimation (default: ");
  print_value ("%d", default_use_polynomial_fit); print_info (")\n");
  print_info ("                     -polynomial_order X = order of the polynomial to be fit (implicitly, use_polynomial_fit = 1) (default: ");
  print_value ("%d", default_polynomial_order); print_info (")\n");
}

//bool
//loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
//{
//  TicToc tt;
//  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

//  tt.tic ();
//  if (loadPCDFile (filename, cloud) < 0)
//    return (false);
//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
//  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

//  return (true);
//}

// from ply2pcd
bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  pcl::PLYReader reader;
  tt.tic ();
  if (reader.read (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

// from normal_estimation.cpp
void
computeNormals (const pcl::PCLPointCloud2 &input, pcl::PCLPointCloud2 &output,
         int k, double radius)
{
  // Convert data to PointCloud<T>
  PointCloud<PointXYZRGB>::Ptr xyz (new PointCloud<PointXYZRGB>);
  fromPCLPointCloud2 (input, *xyz);

  TicToc tt;
  tt.tic ();

  PointCloud<Normal> normals;

  // Try our luck with organized integral image based normal estimation
//  if (xyz->isOrganized ())
//  {
//    IntegralImageNormalEstimation<PointXYZRGB, Normal> ne;
//    ne.setInputCloud (xyz);
//    ne.setNormalEstimationMethod (IntegralImageNormalEstimation<PointXYZRGB, Normal>::COVARIANCE_MATRIX);
//    ne.setNormalSmoothingSize (float (radius));
//    ne.setDepthDependentSmoothing (true);
//    ne.compute (normals);
//  }
//  else
  {
    NormalEstimation<PointXYZRGB, Normal> ne;
    ne.setInputCloud (xyz);
    ne.setSearchMethod (search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));
    ne.setKSearch (k);
    ne.setRadiusSearch (radius);
    ne.setViewPoint( 0.0f,0.0f,1.0f );
    ne.compute (normals);
  }

  print_highlight ("Computed normals in "); print_value ("%g", tt.toc ()); print_info (" ms for "); print_value ("%d", normals.width * normals.height); print_info (" points.\n");

  // Convert data back
  pcl::PCLPointCloud2 output_normals;
  toPCLPointCloud2 (normals, output_normals);
  concatenateFields (input, output_normals, output);
}

void computeMLS (const pcl::PCLPointCloud2::ConstPtr &input, pcl::PCLPointCloud2 &output,
         double search_radius, bool sqr_gauss_param_set, double sqr_gauss_param,
         bool use_polynomial_fit, int polynomial_order)
{

  PointCloud<PointXYZRGBNormal>::Ptr xyz_cloud_pre (new pcl::PointCloud<PointXYZRGBNormal> ()),
      xyz_cloud (new pcl::PointCloud<PointXYZRGBNormal> ());
  fromPCLPointCloud2 (*input, *xyz_cloud_pre);

  // Filter the NaNs from the cloud
  for (size_t i = 0; i < xyz_cloud_pre->size (); ++i)
    if (pcl_isfinite (xyz_cloud_pre->points[i].x))
      xyz_cloud->push_back (xyz_cloud_pre->points[i]);
  xyz_cloud->header = xyz_cloud_pre->header;
  xyz_cloud->height = 1;
  xyz_cloud->width = static_cast<uint32_t> (xyz_cloud->size ());
  xyz_cloud->is_dense = false;
  
  

  PointCloud<PointXYZRGBNormal>::Ptr xyz_cloud_smoothed (new PointCloud<PointXYZRGBNormal> ());

  MovingLeastSquares<PointXYZRGBNormal, PointXYZRGBNormal> mls;
  mls.setInputCloud (xyz_cloud);
  mls.setSearchRadius (search_radius);
  if (sqr_gauss_param_set) mls.setSqrGaussParam (sqr_gauss_param);
  mls.setPolynomialFit (use_polynomial_fit);
  mls.setPolynomialOrder (polynomial_order);

//  mls.setUpsamplingMethod (MovingLeastSquares<PointXYZ, PointNormal>::SAMPLE_LOCAL_PLANE);
//  mls.setUpsamplingMethod (MovingLeastSquares<PointXYZ, PointNormal>::RANDOM_UNIFORM_DENSITY);
//  mls.setUpsamplingMethod (MovingLeastSquares<PointXYZRGBNormal, PointXYZRGBNormal>::VOXEL_GRID_DILATION);
  mls.setUpsamplingMethod (MovingLeastSquares<PointXYZRGBNormal, PointXYZRGBNormal>::NONE);
  mls.setPointDensity (60000 * int (search_radius)); // 300 points in a 5 cm radius
  mls.setUpsamplingRadius (0.0025);
  mls.setUpsamplingStepSize (0.0015);
  mls.setDilationIterations (2);
  mls.setDilationVoxelSize (0.0025f);

  search::KdTree<PointXYZRGBNormal>::Ptr tree (new search::KdTree<PointXYZRGBNormal> ());
  mls.setSearchMethod (tree);
  mls.setComputeNormals (true);
//  mls.setComputeNormals (false);

  PCL_INFO ("Computing smoothed surface and normals with search_radius %f , sqr_gaussian_param %f, polynomial fitting %d, polynomial order %d\n",
            mls.getSearchRadius(), mls.getSqrGaussParam(), mls.getPolynomialFit(), mls.getPolynomialOrder());
  TicToc tt;
  tt.tic ();
  mls.process (*xyz_cloud_smoothed);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", xyz_cloud_smoothed->width * xyz_cloud_smoothed->height); print_info (" points]\n");

  toPCLPointCloud2 (*xyz_cloud_smoothed, output);
}

//void
//saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &output)
//{
//  TicToc tt;
//  tt.tic ();

//  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());

//  pcl::io::savePCDFile (filename, output,  Eigen::Vector4f::Zero (),
//                        Eigen::Quaternionf::Identity (), true);

//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
//}

void
saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &cloud )
{
  bool binary=false;
  bool use_camera=false;

  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());

  pcl::PLYWriter writer;
  writer.write (filename, cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), binary, use_camera);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
}


/* ---[ */
int
main (int argc, char** argv)
{
  print_info ("Moving Least Squares smoothing of a point cloud. For more information, use: %s -h\n", argv[0]);

  if (argc < 2)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd files
  std::vector<int> p_file_indices;
  p_file_indices = parse_file_extension_argument (argc, argv, ".ply");
  if (p_file_indices.size () == 0)
  {
    print_error ("Need one input PLY file.\n");
    return (-1);
  }

  // Command line parsing
  double mls_search_radius = default_search_radius;
  double mls_sqr_gauss_param = default_sqr_gauss_param;
  bool mls_sqr_gauss_param_set = true;
  int mls_polynomial_order = default_polynomial_order;
  bool mls_use_polynomial_fit = default_use_polynomial_fit;

  parse_argument (argc, argv, "-mls_radius", mls_search_radius);
  if (parse_argument (argc, argv, "-mls_sqr_gauss_param", mls_sqr_gauss_param) == -1)
    mls_sqr_gauss_param_set = false;
  if (parse_argument (argc, argv, "-mls_polynomial_order", mls_polynomial_order) != -1 )
    mls_use_polynomial_fit = true;
  parse_argument (argc, argv, "-mls_use_polynomial_fit", mls_use_polynomial_fit);

  // Command line parsing
  int normal_est_k = 0;
  double normal_est_radius = 0.01;
  parse_argument (argc, argv, "-normal_est_k", normal_est_k);
  parse_argument (argc, argv, "-normal_est_radius", normal_est_radius);


  double icp_corespondance_dist = 0.001;
  parse_argument (argc, argv, "-icp_corespondance_dist", icp_corespondance_dist);


  pcl::PCLPointCloud2 output[p_file_indices.size()];
  for ( int idx=0;idx<p_file_indices.size();idx++) {
      // Load the file
      pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
      if (!loadCloud (argv[p_file_indices[ idx ]], *cloud))
        return (-1);




      // Do the smoothing
      pcl::PCLPointCloud2 tmpOutput;
      computeMLS ( cloud, tmpOutput, mls_search_radius, mls_sqr_gauss_param_set, mls_sqr_gauss_param, mls_use_polynomial_fit, mls_polynomial_order);


      // compute normals after MLS

      computeNormals ( tmpOutput , output[idx], normal_est_k, normal_est_radius);

      // register to cloud 0
      pcl::PointCloud<pcl::PointXYZRGBNormal> target;
      pcl::fromPCLPointCloud2(output[0], target);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pTarget = target.makeShared();

      // TODO: loop over normals and reverse
      // if ( dot(@N,set(0,0,-1) ) > 0.0 )
      //    @N*=-1;

      pcl::PointCloud<pcl::PointXYZRGBNormal> source_aligned;
      if (idx>0) {
          pcl::PointCloud<pcl::PointXYZRGBNormal> source;
          pcl::fromPCLPointCloud2(output[idx], source);
          pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pSource = source.makeShared();
          pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
          icp.setMaximumIterations(10000);
          icp.setMaxCorrespondenceDistance ( icp_corespondance_dist ); //1mm
          icp.setRANSACOutlierRejectionThreshold (0.01); //10mm
          icp.setInputSource ( pSource );
          icp.setInputTarget ( pTarget );
          // Start registration process
          icp.align (source_aligned);

          std::cout << argv[p_file_indices[ idx ]] << "has converged:" << icp.hasConverged () << " score: " << icp.getFitnessScore () << std::endl;
          Eigen::Matrix<float, 4, 4> final=icp.getFinalTransformation ();
          std::cout << final(0)  << "," << final(1)  << "," << final(2)  << "," << final(3) << "," << std::endl
                    << final(4)  << "," << final(5)  << "," << final(6)  << "," << final(7) << "," << std::endl
                    << final(8)  << "," << final(9)  << "," << final(10) << "," << final(11) << "," << std::endl
                    << final(12) << "," << final(13) << "," << final(14) << "," << final(15) << std::endl;

      }
      else {
          source_aligned=target;
      }

      pcl::PCLPointCloud2 writePointCloud2;
      pcl::toPCLPointCloud2( source_aligned, writePointCloud2);
      std::stringstream filename;
      filename << argv[p_file_indices[ idx ]] << "_mls_radius_" << mls_search_radius << "_polyfit_" << mls_use_polynomial_fit*mls_polynomial_order << "_normal_est_" << normal_est_k << "_" << normal_est_radius  << "_icp_cd_" << icp_corespondance_dist << "_registered.ply";
      // Save into the second file
      saveCloud ( filename.str() , writePointCloud2 );
      std::cout << "--------------------------------------------------------" << std::endl;
  }
}
