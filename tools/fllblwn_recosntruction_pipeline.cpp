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
#include <pcl/common/geometry.h> // distance
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/segmentation/sac_segmentation.h>


using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

int default_polynomial_order = 2;
bool default_use_polynomial_fit = false;
double default_search_radius = 0.0,
    default_sqr_gauss_param = 0.0;

bool sortPointByX(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
        if (p1.x < p2.x) {
                return true;
        }
        else {
                return false;
        }
}

bool sortPointByXreversed(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
        if (p1.x < p2.x) {
                return false;
        }
        else {
                return true;
        }
}

bool sortPointByZ(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
        if (p1.z < p2.z) {
                return true;
        }
        else {
                return false;
        }
}


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
//  ne.setViewPoint( 0.0f,0.0f,1.0f );
    ne.compute (normals);
  }

  print_highlight ("Computed normals in "); print_value ("%g", tt.toc ()); print_info (" ms for "); print_value ("%d", normals.width * normals.height); print_info (" points.\n");

  // Convert data back
  pcl::PCLPointCloud2 output_normals;
  toPCLPointCloud2 (normals, output_normals);
  concatenateFields (input, output_normals, output);
}

void computeMLS (const pcl::PCLPointCloud2 &input, pcl::PCLPointCloud2 &output,
         double search_radius, bool sqr_gauss_param_set, double sqr_gauss_param,
         bool use_polynomial_fit, int polynomial_order)
{

  PointCloud<PointXYZRGBNormal>::Ptr xyz_cloud_pre (new pcl::PointCloud<PointXYZRGBNormal> ()),
      xyz_cloud (new pcl::PointCloud<PointXYZRGBNormal> ());
  fromPCLPointCloud2 (input, *xyz_cloud_pre);

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
//  mls.setComputeNormals (true);
  mls.setComputeNormals (false);

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
removeOutliers (const pcl::PCLPointCloud2::ConstPtr &input, pcl::PCLPointCloud2 &output,
         std::string method,
         int min_pts, double radius,
         int mean_k, double std_dev_mul, bool negative, bool keep_organized)
{

  PointCloud<PointXYZ>::Ptr xyz_cloud_pre (new pcl::PointCloud<PointXYZ> ()),
                            xyz_cloud (new pcl::PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*input, *xyz_cloud_pre);

  pcl::PointIndices::Ptr removed_indices (new PointIndices),
                         indices (new PointIndices);
  std::vector<int> valid_indices;
  if (keep_organized)
  {
    xyz_cloud = xyz_cloud_pre;
    for (int i = 0; i < int (xyz_cloud->size ()); ++i)
      valid_indices.push_back (i);
  }
  else
    removeNaNFromPointCloud<PointXYZ> (*xyz_cloud_pre, *xyz_cloud, valid_indices);

  TicToc tt;
  tt.tic ();
  PointCloud<PointXYZ>::Ptr xyz_cloud_filtered (new PointCloud<PointXYZ> ());
  if (method == "statistical")
  {
    StatisticalOutlierRemoval<PointXYZ> filter (true);
    filter.setInputCloud (xyz_cloud);
    filter.setMeanK (mean_k);
    filter.setStddevMulThresh (std_dev_mul);
    filter.setNegative (negative);
    filter.setKeepOrganized (keep_organized);
    PCL_INFO ("Computing filtered cloud from %lu points with mean_k %d, std_dev_mul %f, inliers %d ...", xyz_cloud->size (), filter.getMeanK (), filter.getStddevMulThresh (), filter.getNegative ());
    filter.filter (*xyz_cloud_filtered);
    // Get the indices that have been explicitly removed
    filter.getRemovedIndices (*removed_indices);
  }
  else if (method == "radius")
  {
    RadiusOutlierRemoval<PointXYZ> filter (true);
    filter.setInputCloud (xyz_cloud);
    filter.setRadiusSearch (radius);
    filter.setMinNeighborsInRadius (min_pts);
    filter.setNegative (negative);
    filter.setKeepOrganized (keep_organized);
    PCL_INFO ("Computing filtered cloud from %lu points with radius %f, min_pts %d ...", xyz_cloud->size (), radius, min_pts);
    filter.filter (*xyz_cloud_filtered);
    // Get the indices that have been explicitly removed
    filter.getRemovedIndices (*removed_indices);
  }
  else
  {
    PCL_ERROR ("%s is not a valid filter name! Quitting!\n", method.c_str ());
    return;
  }

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", xyz_cloud_filtered->width * xyz_cloud_filtered->height); print_info (" points, %lu indices removed]\n", removed_indices->indices.size ());

  if (keep_organized)
  {
    pcl::PCLPointCloud2 output_filtered;
    toPCLPointCloud2 (*xyz_cloud_filtered, output_filtered);
    concatenateFields (*input, output_filtered, output);
  }
  else
  {
    // Make sure we are addressing values in the original index vector
    for (size_t i = 0; i < removed_indices->indices.size (); ++i)
      indices->indices.push_back (valid_indices[removed_indices->indices[i]]);

    // Extract the indices of the remaining points
    pcl::ExtractIndices<pcl::PCLPointCloud2> ei;
    ei.setInputCloud (input);
    ei.setIndices (indices);
    ei.setNegative (true);
    ei.filter (output);
  }
}

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

  // Command line parsing MLS
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

  // Command line parsing Normals
  int normal_est_k = 0;
  double normal_est_radius = 0.01;
  parse_argument (argc, argv, "-normal_est_k", normal_est_k);
  parse_argument (argc, argv, "-normal_est_radius", normal_est_radius);


  // Command line parsing ICP
  double icp_corespondance_dist = 0.001;
  parse_argument (argc, argv, "-icp_corespondance_dist", icp_corespondance_dist);

  // cms line ransac
  int sac_max_iterations=10000000;
  float sac_distance_threshold=0.0055,
        sac_min_radius=0.0165,
        sac_max_radius=0.02,
        sac_normal_distance_weight=0.005;
 
  parse_argument (argc, argv, "-sac_distance_threshold", sac_distance_threshold );
  parse_argument (argc, argv, "-sac_max_iterations", sac_max_iterations );
  parse_argument (argc, argv, "-sac_min_radius", sac_min_radius );
  parse_argument (argc, argv, "-sac_max_radius", sac_max_radius );
  parse_argument (argc, argv, "-sac_normal_distance_weight", sac_normal_distance_weight );


 
  // Command line parsing Outlier Removal
  std::string method = "radius";
  int min_pts = 100; // 100; // min points need in radius to keep point
  double radius = 0.0125;
  int mean_k = 150;  // 100 might be ok too - not used in radius method
  double std_dev_mul = 0.0;
  int negative = 0;


  parse_argument (argc, argv, "-method", method);
  parse_argument (argc, argv, "-radius", radius);
  parse_argument (argc, argv, "-min_pts", min_pts);
  parse_argument (argc, argv, "-mean_k", mean_k);
  parse_argument (argc, argv, "-std_dev_mul", std_dev_mul);
  parse_argument (argc, argv, "-negative", negative);
  bool keep_organized = find_switch (argc, argv, "-keep_organized");


  pcl::PCLPointCloud2 output[p_file_indices.size()];

  // these 3 clouds will be filled with our sphere centroids and be used for registration later
  // code from here: https://machinelearning1.wordpress.com/2014/02/09/estimate-the-transformation-matrix-between-two-sets-of-points-pcl/

#define SPHERES_NEEDED 6

  pcl::PointCloud<pcl::PointXYZ>::Ptr regPointCloud[3] ;
  for (int cloudIdx=0;cloudIdx<3;cloudIdx++) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmpPtr (new pcl::PointCloud<pcl::PointXYZ> ());
//      pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
    regPointCloud[cloudIdx]=tmpPtr;
    regPointCloud[cloudIdx]->width = SPHERES_NEEDED;
    regPointCloud[cloudIdx]->height = 1;
    regPointCloud[cloudIdx]->is_dense = false;
    regPointCloud[cloudIdx]->resize(regPointCloud[cloudIdx]->width * regPointCloud[cloudIdx]->height);
  }

  for ( int idx=0;idx<p_file_indices.size();idx++) {
      // Load the file
      pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
      if (!loadCloud (argv[p_file_indices[ idx ]], *cloud))
        return (-1);


      // Do the smoothing
      pcl::PCLPointCloud2 tmpOutputWithoutOutliers;
      min_pts=10;
      removeOutliers( cloud, tmpOutputWithoutOutliers, method, min_pts, radius, mean_k, std_dev_mul, negative, keep_organized);



      pcl::PCLPointCloud2 tmpOutputMLS;
      computeMLS ( tmpOutputWithoutOutliers, tmpOutputMLS, mls_search_radius, mls_sqr_gauss_param_set, mls_sqr_gauss_param, mls_use_polynomial_fit, mls_polynomial_order);

      //TODO: compute normals before MLS - they will be preserved and only XYZ will be smoothed

      // compute normals after MLS
      computeNormals ( tmpOutputMLS , output[idx], normal_est_k, normal_est_radius);
//      computeNormals ( tmpOutputWithoutOutliers , output[idx], normal_est_k, normal_est_radius);


      // Detect the largest plane and remove it from the sets
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

      // segmentation
      pcl::PointCloud<pcl::PointXYZRGBNormal> ranSourcePointXYZRGBNormal;
      pcl::fromPCLPointCloud2(output[idx], ranSourcePointXYZRGBNormal);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pSegSource = ranSourcePointXYZRGBNormal.makeShared();

      //extrac Normals
      pcl::PointCloud<pcl::PointXYZ> ransacSourceXYZ;
      pcl::PointCloud<pcl::Normal> ransacSourceNormals;
      ransacSourceXYZ.points.resize(ranSourcePointXYZRGBNormal.size());
      ransacSourceNormals.points.resize(ranSourcePointXYZRGBNormal.size());
      for (size_t i = 0; i < ranSourcePointXYZRGBNormal.points.size(); i++) {
          //XYZ
          ransacSourceXYZ.points[i].x = ranSourcePointXYZRGBNormal.points[i].x;
          ransacSourceXYZ.points[i].y = ranSourcePointXYZRGBNormal.points[i].y;
          ransacSourceXYZ.points[i].z = ranSourcePointXYZRGBNormal.points[i].z;
          //
          ransacSourceNormals.points[i].normal_x = ranSourcePointXYZRGBNormal.points[i].normal_x;
          ransacSourceNormals.points[i].normal_y = ranSourcePointXYZRGBNormal.points[i].normal_y;
          ransacSourceNormals.points[i].normal_z = ranSourcePointXYZRGBNormal.points[i].normal_z;
      }
      pcl::PointCloud<pcl::PointXYZ>::ConstPtr pRanscaSourceXYZ = ransacSourceXYZ.makeShared();
      pcl::PointCloud<pcl::Normal>::ConstPtr pRansacSourceNormals = ransacSourceNormals.makeShared();

      // Create the segmentation object
      pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
      // Optional
      seg.setOptimizeCoefficients (true);
      // Mandatory
      // sphere segmentation
//        seg.setModelType(SACMODEL_SPHERE);
      seg.setModelType(SACMODEL_NORMAL_SPHERE); //
      seg.setMethodType(SAC_RANSAC);
      seg.setMaxIterations( sac_max_iterations );
      seg.setDistanceThreshold( sac_distance_threshold );
      //seg.setRadiusLimits(0.0195f, 0.0205f);
      seg.setRadiusLimits( sac_min_radius, sac_max_radius );
      //seg.setSamplesMaxDist();
      seg.setNormalDistanceWeight( sac_normal_distance_weight );
//      seg.setNormalDistanceWeight( 0.5f * (3.141592654/180) );
      //seg.setEpsAngle(15 / (180/3.141592654));

      std::cout << sac_distance_threshold << " - " << sac_min_radius << " - " << sac_max_radius << " - " << sac_normal_distance_weight << std::endl;

      seg.setInputCloud (pRanscaSourceXYZ);
      seg.setInputNormals(pRansacSourceNormals );

      bool inliersFound=true;

      //will be used inside the while loop
      pcl::ExtractIndices<pcl::PointXYZ> extractXYZ;
      pcl::ExtractIndices<pcl::Normal> extractNormals;

      int sphereNum=0;

      while (inliersFound) {
        seg.segment(*inliers,*coefficients);

        // loop here as long as we have inliers and remove them from the PC after each iteration
        if (inliers->indices.size () == 0)
        {
        std::cout   << "Could not find inliers for sphere " << sphereNum << " in the given dataset.\n";
        inliersFound=false;
        }
        else {
//          PCL_INFO ("FOUND INLIERS: ");
//          std::cout << inliers->indices.size () << std::endl;

            extractXYZ.setInputCloud (pRanscaSourceXYZ);
            extractXYZ.setIndices (inliers);
            extractXYZ.setNegative (true);
            extractNormals.setInputCloud (pRansacSourceNormals);
            extractNormals.setIndices (inliers);
            extractNormals.setNegative (true);

            extractXYZ.filter( ransacSourceXYZ );
            extractNormals.filter( ransacSourceNormals );

            // update pointers to newly filtered pointclouds
            pRanscaSourceXYZ = ransacSourceXYZ.makeShared();
            pRansacSourceNormals = ransacSourceNormals.makeShared();

            seg.setInputCloud(pRanscaSourceXYZ);
            seg.setInputNormals(pRansacSourceNormals);



        }

        if (sphereNum<SPHERES_NEEDED) {
            regPointCloud[idx]->points[sphereNum].x = coefficients->values[0];
            regPointCloud[idx]->points[sphereNum].y = coefficients->values[1];
            regPointCloud[idx]->points[sphereNum].z = coefficients->values[2];
            }

        std::cout   << coefficients->values[0] << " "
                    << coefficients->values[1] << " "
                    << coefficients->values[2] << " "
                    << coefficients->values[3] << " # "
                    << "sphere" << sphereNum << " ("<<inliers->indices.size()<<" inliers):\\"
                    << std::endl;



        sphereNum++;
      }

      if (idx==0) { // sort cloud 0 by X - for correspondence grouping later - found out manually
          std::sort(regPointCloud[idx]->points.begin(), regPointCloud[idx]->points.end(), sortPointByZ);
      }
      else { // sort cloud 1 & 2 by X reversed
          std::sort(regPointCloud[idx]->points.begin(), regPointCloud[idx]->points.end(), sortPointByX);
      }




      // register to cloud 0
      pcl::PointCloud<pcl::PointXYZRGBNormal> target;
      pcl::fromPCLPointCloud2(output[0], target);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pTarget = target.makeShared();

      pcl::PointCloud<pcl::PointXYZRGBNormal> source_aligned;
      pcl::PointCloud<pcl::PointXYZRGBNormal> outputAligned;
      int registration_enabled=true;
      if (idx>0 ) {
          if ( registration_enabled ) {



              //MOVE 1st point of cloud[idx] to first point of cloud[0]

              /* Reminder: how transformation matrices work :

                       |-------> This column is the translation
                | 1 0 0 x |  \
                | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
                | 0 0 1 z |  /
                | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

                METHOD #1: Using a Matrix4f
                This is the "manual" method, perfect to understand but error prone !
              */
//              Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

//              transform_1 (3) = pointCloud[idx]->points[0].x - pointCloud[0]->points[0].x;
//              transform_1 (7) = pointCloud[idx]->points[0].y - pointCloud[0]->points[0].y;
//              transform_1 (11) = pointCloud[idx]->points[0].z - pointCloud[0]->points[0].z;

//              Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
//              transform_2.translation() <<  regPointCloud[idx]->points[0].x - regPointCloud[0]->points[0].x,
//                                            regPointCloud[idx]->points[0].y - regPointCloud[0]->points[0].y,
//                                            regPointCloud[idx]->points[0].z - regPointCloud[0]->points[0].z;




//                pcl::Correspondence corr0 (0,0,1);pcl::Correspondence corr1 (1,1,1);pcl::Correspondence corr2 (2,2,1);
                boost::shared_ptr<pcl::Correspondences> correspondences (new pcl::Correspondences);
//                pcl::registration::CorrespondenceEstimation<PointXYZ, PointXYZ> corr_est;
//                corr_est.setInputSource (pointCloud[0]);
//                corr_est.setInputTarget (pointCloud[idx]);
//                corr_est.determineCorrespondences(*correspondences);
                correspondences->push_back( pcl::Correspondence(0,0, pcl::geometry::distance(regPointCloud[idx]->points[0],regPointCloud[0]->points[0]) ) );
                correspondences->push_back( pcl::Correspondence(2,1, pcl::geometry::distance(regPointCloud[idx]->points[1],regPointCloud[0]->points[2]) ) );
                correspondences->push_back( pcl::Correspondence(1,2, pcl::geometry::distance(regPointCloud[idx]->points[2],regPointCloud[0]->points[1]) ) );
                correspondences->push_back( pcl::Correspondence(3,4, pcl::geometry::distance(regPointCloud[idx]->points[3],regPointCloud[0]->points[4]) ) );
                correspondences->push_back( pcl::Correspondence(4,3, pcl::geometry::distance(regPointCloud[idx]->points[4],regPointCloud[0]->points[3]) ) );
                correspondences->push_back( pcl::Correspondence(5,5, pcl::geometry::distance(regPointCloud[idx]->points[5],regPointCloud[0]->points[5]) ) );


                // https://machinelearning1.wordpress.com/2014/02/09/estimate-the-transformation-matrix-between-two-sets-of-points-pcl/
                pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ> TESVD;
                pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ>::Matrix4 rigidTransformSVD;
                TESVD.estimateRigidTransformation (*regPointCloud[idx],*regPointCloud[0],*correspondences,rigidTransformSVD);
//                TESVD.estimateRigidTransformation (*regPointCloud[idx],*regPointCloud[0],transformation2);


//                pcl::registration::TransformationEstimation3Point<pcl::PointXYZ,pcl::PointXYZ> TE3P;
//                pcl::registration::TransformationEstimation3Point<pcl::PointXYZ,pcl::PointXYZ>::Matrix4 transformation2;
//                TE3P.estimateRigidTransformation (*pointCloud[0],*pointCloud[idx],*correspondences,transformation2);
//                TE3P.estimateRigidTransformation (*pointCloud[0],*transformed_cloud,*correspondences,transformation2);


                for (int debugIdx=0;debugIdx<SPHERES_NEEDED;debugIdx++) {
//                    std::cout << regPointCloud[idx]->points[debugIdx].x << " X " << regPointCloud[0]->points[debugIdx].x << std::endl <<
//                                 regPointCloud[idx]->points[debugIdx].y << " Y " << regPointCloud[0]->points[debugIdx].y << std::endl <<
//                                 regPointCloud[idx]->points[debugIdx].z << " Z " << regPointCloud[0]->points[debugIdx].z << std::endl;

                    printf ("%6.3f %6.3f \n", regPointCloud[idx]->points[debugIdx].x, regPointCloud[0]->points[debugIdx].x );
                    printf ("%6.3f %6.3f \n", regPointCloud[idx]->points[debugIdx].y, regPointCloud[0]->points[debugIdx].y );
                    printf ("%6.3f %6.3f \n", regPointCloud[idx]->points[debugIdx].z, regPointCloud[0]->points[debugIdx].z );
                    printf ("\n");
                }

//                std::cout << "The Estimated Rotation and translation matrices (using getTransformation function) are : \n" << std::endl;
//                printf ("\n");
//                printf ("    | %6.3f %6.3f %6.3f | \n", transformation2 (0,0), transformation2 (0,1), transformation2 (0,2));
//                printf ("R = | %6.3f %6.3f %6.3f | \n", transformation2 (1,0), transformation2 (1,1), transformation2 (1,2));
//                printf ("    | %6.3f %6.3f %6.3f | \n", transformation2 (2,0), transformation2 (2,1), transformation2 (2,2));
//                printf ("\n");
//                printf ("t = < %0.3f, %0.3f, %0.3f >\n", transformation2 (0,3), transformation2 (1,3), transformation2 (2,3));
//                printf ("\n");
//                printf ("%6.3f %6.3f %6.3f %6.3f  \n", transformation2 (0,0), transformation2 (0,1), transformation2 (0,2), transformation2 (0,3));
//                printf ("%6.3f %6.3f %6.3f %6.3f  \n", transformation2 (1,0), transformation2 (1,1), transformation2 (1,2), transformation2 (1,3));
//                printf ("%6.3f %6.3f %6.3f %6.3f  \n", transformation2 (2,0), transformation2 (2,1), transformation2 (2,2), transformation2 (2,3));
//                printf ("%6.3f %6.3f %6.3f %6.3f  \n", transformation2 (3,0), transformation2 (3,1), transformation2 (3,2), transformation2 (3,3));
                  std::cout << rigidTransformSVD(0)  << "," << rigidTransformSVD(1)  << "," << rigidTransformSVD(2)  << "," << rigidTransformSVD(3) << "," << std::endl
                            << rigidTransformSVD(4)  << "," << rigidTransformSVD(5)  << "," << rigidTransformSVD(6)  << "," << rigidTransformSVD(7) << "," << std::endl
                            << rigidTransformSVD(8)  << "," << rigidTransformSVD(9)  << "," << rigidTransformSVD(10) << "," << rigidTransformSVD(11) << "," << std::endl
                            << rigidTransformSVD(12) << "," << rigidTransformSVD(13) << "," << rigidTransformSVD(14) << "," << rigidTransformSVD(15) << std::endl;



                  // apply trasnsform before ICP
                  pcl::PointCloud<pcl::PointXYZRGBNormal> icpSource, icpTarget;
                  pcl::fromPCLPointCloud2(output[idx], icpSource);
                  pcl::fromPCLPointCloud2(output[0], icpTarget);
                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr icpSourceRigidTransformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
                  pcl::transformPointCloud ( icpSource, *icpSourceRigidTransformed, rigidTransformSVD );

                    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr pIcpTarget = icpTarget.makeShared();
                    // pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
                    pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
                    icp.setMaximumIterations(1000000);
                    icp_corespondance_dist=0.01;
                    icp.setMaxCorrespondenceDistance ( icp_corespondance_dist ); //1mm
                    icp.setRANSACOutlierRejectionThreshold (1.0); //10mm
                    icp.setInputSource ( icpSourceRigidTransformed ); // ( pSource );
                    icp.setInputTarget ( pIcpTarget );//( pTarget );
                    // Start registration process
                    icp.align (outputAligned);

                    std::cout << argv[p_file_indices[ idx ]] << "has converged:" << icp.hasConverged () << " score: " << icp.getFitnessScore () << std::endl;
                    Eigen::Matrix<float, 4, 4> finalIcpTransform=icp.getFinalTransformation ();
                    std::cout << finalIcpTransform(0)  << "," << finalIcpTransform(1)  << "," << finalIcpTransform(2)  << "," << finalIcpTransform(3) << "," << std::endl
                              << finalIcpTransform(4)  << "," << finalIcpTransform(5)  << "," << finalIcpTransform(6)  << "," << finalIcpTransform(7) << "," << std::endl
                              << finalIcpTransform(8)  << "," << finalIcpTransform(9)  << "," << finalIcpTransform(10) << "," << finalIcpTransform(11) << "," << std::endl
                              << finalIcpTransform(12) << "," << finalIcpTransform(13) << "," << finalIcpTransform(14) << "," << finalIcpTransform(15) << std::endl;

              //pcl::fromPCLPointCloud2( output[idx], outputAligned );






          }
          else {
            pcl::fromPCLPointCloud2( output[idx], outputAligned );
          }

      }
      else {
          outputAligned=target;
      }

      pcl::PCLPointCloud2 writePointCloud2;
      pcl::toPCLPointCloud2( outputAligned, writePointCloud2);
      std::stringstream filename;
      filename << argv[p_file_indices[ idx ]] << "_mls_radius_" << mls_search_radius << "_polyfit_" << mls_use_polynomial_fit*mls_polynomial_order << "_normal_est_" << normal_est_k << "_" << normal_est_radius  << "_icp_cd_" << icp_corespondance_dist << "_reg_enabled_" << registration_enabled << "_outliersRemoved.ply";
      // Save into the second file
      saveCloud ( filename.str() , writePointCloud2 );
      std::cout << "--------------------------------------------------------" << std::endl;
  }
}
