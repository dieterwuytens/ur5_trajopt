#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <jsoncpp/json/json.h>
#include <ros/ros.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <tesseract_ros/kdl/kdl_chain_kin.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <trajopt/plot_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_utils/config.hpp>
#include <trajopt_utils/logging.hpp>

#include <visualization_msgs/Marker.h>

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

using namespace trajopt;
using namespace tesseract;

const std::string ROBOT_DESCRIPTION_PARAM = "robot_description"; //< Default ROS parameter for robot description 
const std::string ROBOT_SEMANTIC_PARAM = "robot_description_semantic"; //< Default ROS parameter for robot
                                                                         
const std::string TRAJOPT_DESCRIPTION_PARAM =
    "trajopt_description"; //< Default ROS parameter for trajopt description

const double pi = 3.141592653589793238463;

// declaratie voor de spuitkop
double hoek_spuitkop_dgr = 22.6; 					                      // [graden]
// voorgedifineerde afstand is  {0.1; 0.2; 0.25; 0.3; 0.4; 0.5}
// voorgedifineerde hoeken zijn {22.6; 43.6; 53.1; 62; 77.3; 90}

double hoek_spuitkop_rad = hoek_spuitkop_dgr * pi / 180;
double afstand_spuitkop_tot_muur = 0.25; 			              // [meter]
double overlap_procent = 0.5;	// nooit op 100% zetten       // [procent]
double b = 0.4;
double h = 0.3;

// omv voor spuitkop 
double h_spuitkop = (afstand_spuitkop_tot_muur * tan(hoek_spuitkop_rad/2)) * ((1-overlap_procent)*2);

int aantal_banen = round(h/h_spuitkop);  //afronden naar boven tot een integer
double h_spuitkop_overlap = h/aantal_banen;

// declaratie voor aantal stappen in traject
int steps_voor_y = 100;

double delta_y = b / steps_voor_y;
int steps_voor_x = round(h_spuitkop_overlap / delta_y);
double delta_x =  h_spuitkop_overlap / steps_voor_x;

static int steps_ = (aantal_banen) * (steps_voor_y + steps_voor_x);

// declaratie voor trajopt
static bool plotting_ = false;
static float dt = 1.0/steps_;
static std::string method_ = "cpp";
static urdf::ModelInterfaceSharedPtr urdf_model_; //< URDF Model 
static srdf::ModelSharedPtr srdf_model_;          //< SRDF Model 
static tesseract_ros::KDLEnvPtr env;             //< Trajopt Basic Environment 

void addCollisionObject(){
  AttachableObjectPtr obj(new AttachableObject());
  std::shared_ptr<shapes::Mesh>
  visual_mesh(shapes::createMeshFromResource("package://trajopt_examples/meshes/collisionobject_acro.dae"));
  std::shared_ptr<shapes::Mesh>
  collision_mesh(shapes::createMeshFromResource("package://trajopt_examples/meshes/collisionobject_acro.stl"));
  Eigen::Isometry3d obj_pose;
  obj_pose.setIdentity();

  obj->name = "collisionObj";
  obj->visual.shapes.push_back(visual_mesh);
  obj->visual.shape_poses.push_back(obj_pose);
  obj->collision.shapes.push_back(collision_mesh);
  obj->collision.shape_poses.push_back(obj_pose);
  obj->collision.collision_object_types.push_back(CollisionObjectType::ConvexHull);

  env->addAttachableObject(obj);

  AttachedBodyInfo attached_body;
  attached_body.object_name = obj->name;
  attached_body.parent_link_name = "base_link";
  attached_body.transform.setIdentity();
  attached_body.transform.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  //attached_body.touch_links = {"shoulder_link", "upper_arm_link", "forearm_link",
  // "wrist_1_link", "wrist_2_link", "wrist_3_link"};

  env->attachBody(attached_body);
}

TrajOptProbPtr cppMethod()
{
  ProblemConstructionInfo pci(env);

  // Populate Basic Info
  pci.basic_info.n_steps = steps_;
  pci.basic_info.manip = "manipulator";
  pci.basic_info.start_fixed = false;
  pci.basic_info.use_time = false;
  //  pci.basic_info.dofs_fixed

  // Create Kinematic Object
  pci.kin = pci.env->getManipulator(pci.basic_info.manip);

  // Populate Init Info
  Eigen::VectorXd start_pos = pci.env->getCurrentJointValues(pci.kin->getName());

  pci.init_info.type = InitInfo::STATIONARY;
  pci.init_info.data = start_pos.transpose().replicate(pci.basic_info.n_steps, 1);

  // Populate Cost Info
  std::shared_ptr<JointVelTermInfo> jv = std::shared_ptr<JointVelTermInfo>(new JointVelTermInfo);
  jv->coeffs = std::vector<double>(6, 5.0);
  jv->targets = std::vector<double>(6, 0.0);
  jv->first_step = 0;
  jv->last_step = pci.basic_info.n_steps - 1;
  jv->name = "joint_vel";
  jv->term_type = TT_COST;
  pci.cost_infos.push_back(jv);

  std::shared_ptr<CollisionTermInfo> collision = std::shared_ptr<CollisionTermInfo>(new CollisionTermInfo);
  collision->name = "collision";
  collision->term_type = TT_COST;
  collision->continuous = false;
  collision->first_step = 0;
  collision->last_step = pci.basic_info.n_steps - 1;
  collision->gap = 1;
  collision->info = createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 20);
  //pci.cost_infos.push_back(collision);

  int waypoint = 0;

  double x = 0.65586677727;
  double y = 0.109246185391;
  double z = 0.0839935774293;

  double ox = -1.0;
  double oy = 0.0;
  double oz = 0.0;
  double ow = 0.0;

  int timestep_counter = 0;
  bool min_of_plus_y = false;

  // Populate Constraint
  std::shared_ptr<CartPoseTermInfo> pose = std::shared_ptr<CartPoseTermInfo>(new CartPoseTermInfo);
  pose->term_type = TT_CNT;
  pose->name = "waypoint_cart_0";
  pose->link = "ee_link";
  pose->timestep = 0;

  pose->xyz = Eigen::Vector3d(x-0.2, y, z);
  pose->wxyz = Eigen::Vector4d(ow, ox, oy, oz);
  pose->pos_coeffs = Eigen::Vector3d(10, 10, 10);
  pose->rot_coeffs = Eigen::Vector3d(10, 10, 10);
  pci.cnt_infos.push_back(pose);

  std::shared_ptr<CartPoseTermInfo> pose2 = std::shared_ptr<CartPoseTermInfo>(new CartPoseTermInfo);
  pose2->term_type = TT_CNT;
  pose2->name = "waypoint_cart_1";
  pose2->link = "ee_link";
  pose2->timestep = 1;

  pose2->xyz = Eigen::Vector3d(x-0.2, y-0.4, z);
  pose2->wxyz = Eigen::Vector4d(ow, ox, oy, oz);
  pose2->pos_coeffs = Eigen::Vector3d(10, 10, 10);
  pose2->rot_coeffs = Eigen::Vector3d(10, 10, 10);
  pci.cnt_infos.push_back(pose2);

  for (auto w = 0; w < aantal_banen*2; ++w) {
    if ( w % 2 == 0){ 
      // even getallen --> beweeg in y richting

      for (auto i = 0; i < steps_voor_y; ++i) {
      std::shared_ptr<CartPoseTermInfo> pose = std::shared_ptr<CartPoseTermInfo>(new CartPoseTermInfo);
      pose->term_type = TT_CNT;
      pose->name = "waypoint_cart_" + std::to_string(w) + "_" + std::to_string(i);
      pose->link = "ee_link";
      pose->timestep = timestep_counter;

      timestep_counter++;

      if (min_of_plus_y == false){
        y -= delta_y;
      } else {
        y += delta_y;
      }

      pose->xyz = Eigen::Vector3d(x, y, z);
      pose->wxyz = Eigen::Vector4d(ow, ox, oy, oz);
      pose->pos_coeffs = Eigen::Vector3d(10, 10, 10);
      pose->rot_coeffs = Eigen::Vector3d(10, 10, 10);
      pci.cnt_infos.push_back(pose);
      }

      // min/plus veranderen
      if (min_of_plus_y == false){
        min_of_plus_y = true;
      } else {
        min_of_plus_y = false;
      }

    } else {
      // oneven getallen --> beweeg in z richting

      for (auto i = 0; i < steps_voor_x; ++i) {
      std::shared_ptr<CartPoseTermInfo> pose = std::shared_ptr<CartPoseTermInfo>(new CartPoseTermInfo);
      pose->term_type = TT_CNT;
      pose->name = "waypoint_cart_" + std::to_string(w) + "_" + std::to_string(i);
      pose->link = "ee_link";
      pose->timestep = timestep_counter;

      if (w != (aantal_banen*2)-1){
        timestep_counter++;
        x -= delta_x;
      }
      
      pose->xyz = Eigen::Vector3d(x, y, z);
      pose->wxyz = Eigen::Vector4d(ow, ox, oy, oz);
      pose->pos_coeffs = Eigen::Vector3d(10, 10, 10);
      pose->rot_coeffs = Eigen::Vector3d(10, 10, 10);
      pci.cnt_infos.push_back(pose);
      }

    }
  }

  return ConstructProblem(pci);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ur5 case");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;

  // Initial setup
  std::string urdf_xml_string, srdf_xml_string;
  nh.getParam(ROBOT_DESCRIPTION_PARAM, urdf_xml_string);
  nh.getParam(ROBOT_SEMANTIC_PARAM, srdf_xml_string);
  urdf_model_ = urdf::parseURDF(urdf_xml_string);

  srdf_model_ = srdf::ModelSharedPtr(new srdf::Model);
  srdf_model_->initString(*urdf_model_, srdf_xml_string);
  env = tesseract_ros::KDLEnvPtr(new tesseract_ros::KDLEnv);
  assert(urdf_model_ != nullptr);
  assert(env != nullptr);

  bool success = env->init(urdf_model_, srdf_model_);
  assert(success);

  // publisher aanmaken
  ros::Publisher vis_pub = nh.advertise<visualization_msgs::Marker>( "visualization_marker2", 0 );

  // controle overlap moet altijd < 100%
  if (overlap_procent >= 1.0){
    ROS_INFO_STREAM("ERROR: Overlap is 100 procent");
    ROS_INFO("Done");
    return 0;
  }

  // Create plotting tool
  tesseract_ros::ROSBasicPlottingPtr plotter(new tesseract_ros::ROSBasicPlotting(env));

  // Get ROS Parameters
  pnh.param("plotting", plotting_, plotting_);
  pnh.param<std::string>("method", method_, method_);
  pnh.param<int>("steps", steps_, steps_);

  ROS_INFO_STREAM("Ingestelde hoek van spuitkop is " << hoek_spuitkop_dgr << " graden.");
  ROS_INFO_STREAM("Afstand tot de muur is ingesteld op " << afstand_spuitkop_tot_muur << " meter.");
  ROS_INFO_STREAM("Overlap van zigzag baan is ingesteld op " << overlap_procent*100 << " procent.");
  ROS_INFO_STREAM("Aantal banen = " << aantal_banen);
  ROS_INFO_STREAM("Hoogte spuitkop = " << h_spuitkop_overlap);
  ROS_INFO_STREAM("Aantal steps = " << steps_);

  ROS_INFO_STREAM("CHECK URDF --> HOOGTE SPUITKOP = " << h_spuitkop*2);
  ros::Duration(3.5).sleep();

  // open file voor output van time
  std::ofstream time_file;
  time_file.open("/home/dieter/Desktop/trajoptberekeningen/time.csv");
  time_file << dt;
  time_file << "\n";
  time_file << steps_;
  time_file << "\n";
  time_file.close();

  // open file voor output van array
  std::ofstream trajopt_file;
  trajopt_file.open("/home/dieter/Desktop/trajoptberekeningen/pos.csv");

  // Set the robot initial state
  std::unordered_map<std::string, double> ipos;
  ipos["shoulder_pan_joint"] = 0.0;
  ipos["shoulder_lift_joint"] = -0.8976;
  ipos["elbow_joint"] = 1.57;
  ipos["wrist_1_joint"] = -0.6905;
  ipos["wrist_2_joint"] = 1.57;
  ipos["wrist_3_joint"] = 0.0;

  env->setState(ipos);

  plotter->plotScene();

  // Set Log Level
  util::gLogLevel = util::LevelInfo;

  // setup collision object
  //addCollisionObject();

  // Setup Problem
  TrajOptProbPtr prob;
  prob = cppMethod();
  
  // Solve Trajectory
  ROS_INFO("UR5 case");

  std::vector<tesseract::ContactResultMap> collisions;
  ContinuousContactManagerBasePtr manager = prob->GetEnv()->getContinuousContactManager();
  manager->setActiveCollisionObjects(prob->GetKin()->getLinkNames());
  manager->setContactDistanceThreshold(0);

  bool found = tesseract::continuousCollisionCheckTrajectory(
      *manager, *prob->GetEnv(), *prob->GetKin(), prob->GetInitTraj(), collisions);

  ROS_INFO((found) ? ("Final trajectory is in collision") : ("Final trajectory is collision free"));

  sco::BasicTrustRegionSQP opt(prob);

  opt.initialize(trajToDblVec(prob->GetInitTraj()));
  opt.optimize();

  collisions.clear();
  found = tesseract::continuousCollisionCheckTrajectory(
      *manager, *prob->GetEnv(), *prob->GetKin(), prob->GetInitTraj(), collisions);

  ROS_INFO((found) ? ("Final trajectory is in collision") : ("Final trajectory is collision free"));

  TrajArray output = getTraj(opt.x(), prob->GetVars()); 
  ROS_INFO_STREAM("\n" << output );

  // wegschrijven in file
  for(int k=0; k<=steps_-1; k++){
    trajopt_file << "-:";
    for(int r=0; r<5; r++){
      trajopt_file << output(k,r);
      trajopt_file << ",";
    }
    trajopt_file << output(k,5);
    trajopt_file << "\n";
  }

  trajopt_file << "end: \n";
  trajopt_file.close();

  ROS_INFO_STREAM("Wegschrijven data is gelukt.");
  // -------------------------------------------------------------------
  // omvormen data naar xyz 
  system("rosrun moveit_tutorials convert_ur5_trajopt_to_excel");
  ROS_INFO_STREAM("Omvorming data naar xyz is gelukt.");

  // -------------------------------------------------------------------
  // teken geverfd gebied en plot trajectory

  // declaratie vars 
  string x;
  string y;
  string z;

  double double_x;
  double double_y;
  double double_z;

  // array voor markers
  std::vector<visualization_msgs::Marker> arrayMarkers1;
  std::vector<visualization_msgs::Marker> arrayMarkers2;

  int total_steps = steps_voor_y + steps_voor_x;

  for(int r=0; r<=20; r++){

    // -------------------------------------------------------------------
    // plot trajectory
    plotter->plotTrajectory(prob->GetKin()->getJointNames(), getTraj(opt.x(), prob->GetVars()));
    ROS_INFO_STREAM(" -- TRAJECTORY IS AAN HET SIMULEREN --");
    
    // -------------------------------------------------------------------

    ROS_INFO_STREAM(" -- Inlezen van pos en omvormen naar markers --"); 

    int id = 1;

    // inlezen bestanden
    ifstream ixyz("/home/dieter/Desktop/trajoptberekeningen/xyz_correct.csv");
    if(!ixyz.is_open()) std::cout << "ERROR: File Open" << '\n';

    while (ixyz.good()) {

      if (h_spuitkop_overlap <= 0.0){
        ROS_INFO_STREAM("ERROR HOOGTE SPUITKOP IS 0"); 
      }

      // inlezen xyz
      getline(ixyz,x,',');
      getline(ixyz,y,',');
      getline(ixyz,z,'\n');

      // omvormen
      try {
        double_x = std::stod(x);
        double_y = std::stod(y);
        double_z = std::stod(z);
      } catch (const std::invalid_argument&) {
        ROS_INFO_STREAM("ERROR OMVORMING STRING --> DOUBLE"); 
        break;
      }

      // -------------------------------------------------------------------------
      // markes aanmaken
      visualization_msgs::Marker marker1;
      marker1.header.frame_id = "base_link";
      marker1.header.stamp = ros::Time();
      marker1.ns = "my_namespace";
      marker1.id = id;
      id++;
      marker1.type = visualization_msgs::Marker::CUBE;
      marker1.action = visualization_msgs::Marker::ADD;

      // positie xyz
      marker1.pose.position.x = double_x;
      marker1.pose.position.y = double_y;
      marker1.pose.position.z = double_z + h_spuitkop_overlap/2;

      // orientatie wxyz
      marker1.pose.orientation.x = 0.0;
      marker1.pose.orientation.y = 0.0;
      marker1.pose.orientation.z = 0.0;
      marker1.pose.orientation.w = 1.0;

      // schaal
      marker1.scale.x = 0.01;
      marker1.scale.y = 0.01;
      marker1.scale.z = h_spuitkop_overlap;

      // kleur
      marker1.color.a = 0.4;
      marker1.color.r = 1.0;
      marker1.color.g = 0.0;
      marker1.color.b = 0.0;

      marker1.lifetime = ros::Duration(60.0);

      // marker opslaan in array en publisen
      arrayMarkers1.push_back(marker1);
      vis_pub.publish(marker1);

      // ----------------------------------------------------------
      // marker 2

      visualization_msgs::Marker marker2;
      marker2 = marker1;
      marker2.id = id;
      marker2.pose.position.z = double_z - h_spuitkop_overlap/2;

      if (id/2 >= total_steps){
        // alle andere banen mogen feller rood zijn
        marker2.color.a = 1.0;
        marker2.color.r = 1.0;
        marker2.color.g = 0.0;
        marker2.color.b = 0.0;
      } 

      arrayMarkers2.push_back(marker2);
      vis_pub.publish(marker2);

      id++;

      // eventjes wachten met dt
      float sleep = 0.03195;
      ros::Duration(sleep).sleep();
    }

    ixyz.close();

    // markers verwijderen
    ROS_INFO_STREAM("----------- MARKER VERWIJDEREN --------------");
    int markerTeller = 0;
    for (auto m : arrayMarkers1) {
      m.action = visualization_msgs::Marker::DELETE;
      arrayMarkers2[markerTeller].action = visualization_msgs::Marker::DELETE;
      vis_pub.publish(m);
      vis_pub.publish(arrayMarkers2[markerTeller]);
      ros::Duration(0.001).sleep(); // default: 0.001
      markerTeller++;
    }

    ros::Duration(5.0).sleep();
  }
  // -------------------------------------------------------------------

  ROS_INFO("Done");
  ros::spin();
  return 0;
}
