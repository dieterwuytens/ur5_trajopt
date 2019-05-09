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

#include <tesseract_core/basic_types.h>
#include <tesseract_planning/trajopt/trajopt_planner.h>

#include<moveit/move_group_interface/move_group.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

#include <moveit/trajectory_processing/iterative_time_parameterization.h>

#include <trajopt/file_write_callback.hpp>
#include <trajopt/plot_callback.hpp>
#include <trajopt_utils/logging.hpp>

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

// --------------------------------------------------------------------------------------------------------------

trajectory_msgs::JointTrajectory trajArrayToJointTrajectoryMsg(std::vector<std::string> joint_names,
                                                               tesseract::TrajArray traj_array,
                                                               bool use_time,
                                                               ros::Duration time_increment)
{
  // Create the joint trajectory
  trajectory_msgs::JointTrajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.header.frame_id = "0";
  traj_msg.joint_names = joint_names;

  tesseract::TrajArray pos_mat;
  tesseract::TrajArray time_mat;
  if (use_time)
  {
    // Seperate out the time data in the last column from the joint position data
    pos_mat = traj_array.leftCols(traj_array.cols());
    time_mat = traj_array.rightCols(1);
  }
  else
  {
    pos_mat = traj_array;
  }

  ros::Duration time_from_start(0);
  for (int ind = 0; ind < traj_array.rows(); ind++)
  {
    // Create trajectory point
    trajectory_msgs::JointTrajectoryPoint traj_point;

    // Set the position for this time step
    auto mat = pos_mat.row(ind);
    std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    traj_point.positions = vec;

    // Add the current dt to the time_from_start
    if (use_time)
    {
      time_from_start += ros::Duration(time_mat(ind, time_mat.cols() - 1));
    }
    else
    {
      time_from_start += time_increment;
    }
    traj_point.time_from_start = time_from_start;

    traj_msg.points.push_back(traj_point);
  }
  return traj_msg;
}

// ----------------------------------------------------------------------------------------------------------------

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
  pci.basic_info.use_time = true;
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
  jv->term_type = TT_CNT | TT_USE_TIME;;
  pci.cost_infos.push_back(jv);

  std::shared_ptr<CollisionTermInfo> collision = std::shared_ptr<CollisionTermInfo>(new CollisionTermInfo);
  collision->name = "collision";
  collision->term_type = TT_CNT | TT_USE_TIME;;
  collision->continuous = false;
  collision->first_step = 0;
  collision->last_step = pci.basic_info.n_steps - 1;
  collision->gap = 1;
  collision->info = createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 20);
  //pci.cost_infos.push_back(collision);

  int waypoint = 0;

  double x = 0.590001653084;
  double y = 0.12014305211;
  double z = 0.158311145042;

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
  ipos["shoulder_pan_joint"] = 0.772991120815;
  ipos["shoulder_lift_joint"] = -1.12945110003;
  ipos["elbow_joint"] = 1.76839208603;
  ipos["wrist_1_joint"] = -2.22183162371;
  ipos["wrist_2_joint"] = 4.71922874451;
  ipos["wrist_3_joint"] = -0.0558503309833;

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

  // Execute trajectory
  std::cout << "Execute Trajectory on hardware? y/n \n";
  char input = 'n';
  std::cin >> input;
  if (input == 'y')
  {
    std::cout << "Executing... \n";

    // Create action client to send trajectories
    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> execution_client("follow_joint_trajectory", true);
    execution_client.waitForServer();

    // Convert TrajArray (Eigen Matrix of joint values) to ROS message
    trajectory_msgs::JointTrajectory traj_msg;
    ros::Duration t(0.25);

    std::vector<std::string> joint_names;
    joint_names.push_back("shoulder_pan_joint");
    joint_names.push_back("shoulder_lift_joint");
    joint_names.push_back("elbow_joint");
    joint_names.push_back("wrist_1_joint");
    joint_names.push_back("wrist_2_joint");
    joint_names.push_back("wrist_3_joint");

    traj_msg = trajArrayToJointTrajectoryMsg(joint_names, output, false, t);

    // traj_msg bevat nu enkel joint posities, UR5 driver verwacht ook joint velocities dus die gaan we hier toevoegen:
    // --------------------------------------------------------------------------------------------
    //moveit_msgs::RobotTrajectory trajectory_msg;
  

    // First to create a RobotTrajectory object
    moveit::planning_interface::MoveGroup group("manipulator");

    group.setPlanningTime(20.0);
    robot_trajectory::RobotTrajectory rt(group.getCurrentState()->getRobotModel(), "manipulator");
    std::string eef_link = group.getEndEffectorLink();
    std::string eef = group.getEndEffector();
    std::string base_link = group.getPoseReferenceFrame();
    group.setStartStateToCurrentState();
    group.setPoseReferenceFrame(base_link);
    group.setEndEffectorLink(eef_link);


    // Second get a RobotTrajectory from trajectory
    rt.setRobotTrajectoryMsg(*group.getCurrentState(), traj_msg);
  
    // Thrid create a IterativeParabolicTimeParameterization object
    trajectory_processing::IterativeParabolicTimeParameterization iptp;

    // Fourth compute computeTimeStamps
    success = iptp.computeTimeStamps(rt);
    ROS_INFO("Computed time stamp %s",success?"SUCCEDED":"FAILED");

    // Get RobotTrajectory_msg from RobotTrajectory
    //moveit_msgs::RobotTrajectory trajectory_msg;
    rt.getRobotTrajectoryMsg(traj_msg);

    /*
    // Finally plan and execute the trajectory
    plan.trajectory_ = trajectory_msg;
    ROS_INFO("Visualizing plan 4 (cartesian path) (%.2f%% acheived)",fraction * 100.0);   
    sleep(5.0);
    group.execute(plan);
    */

    // --------------------------------------------------------------------------------------------
    // Create action message
    control_msgs::FollowJointTrajectoryGoal trajectory_action;
    trajectory_action.trajectory = traj_msg;
    //        trajectory_action.trajectory.header.frame_id="world";
    //        trajectory_action.trajectory.header.stamp = ros::Time(0);
    //        trajectory_action.goal_time_tolerance = ros::Duration(1.0);
    // May need to update other tolerances as well.

    // Send to hardware
    execution_client.sendGoal(trajectory_action);
    execution_client.waitForResult(ros::Duration(20.0));

    if (execution_client.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      std::cout << "Pick action succeeded! \n";
    }
    else
    {
      std::cout << "Pick action failed \n";
    }
  }

  ROS_INFO("Done");
  ros::spin();
  return 0;
}
