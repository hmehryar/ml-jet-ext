#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include <iostream> // needed for io
#include <cstdio>   // needed for io
#include <fstream>
#include <cmath>
#include <iomanip> // For setw function
#include <string>
#include <sstream>

using namespace std;
using namespace fastjet;

// add near the top with other includes
#include <dirent.h>     // opendir/readdir
#include <sys/stat.h>   // stat, mkdir
#include <vector>
#include <algorithm>

// --- helpers ---
static bool is_dir(const std::string& p) {
  struct stat st; return ::stat(p.c_str(), &st)==0 && S_ISDIR(st.st_mode);
}
static void ensure_dir(const std::string& p) {
  struct stat st;
  if (::stat(p.c_str(), &st)==0 && S_ISDIR(st.st_mode)) return;
  ::mkdir(p.c_str(), 0775);
}
static std::string basename_of(const std::string& path) {
  auto pos = path.find_last_of('/');
  return (pos == std::string::npos) ? path : path.substr(pos+1);
}


//------------------------------------------------------------------------
// the user information

class MyInfo : public PseudoJet::UserInfoBase{
public:
  // default ctor
  //  - pdg_id        the PDG id of the particle
  MyInfo(const float status_id,const float pid){
    _status_id=status_id;
    _pid=pid;
  }
  MyInfo(const MyInfo& other) : _pid(other._pid), _status_id(other._status_id) {};

  /// access to the PDG id
  float status_id() const { return _status_id;}
  float pid() const {return _pid;}
  
protected:
  float _status_id;         // the associated status id
  float _pid;               //assosciated pid
};

//----------------------------------------------
//Subtracts Holes
  //check if the particle is charged
bool Check_Charged(float pid){
	// Charged hadrons: (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-) 
	float charged_hadrons_pid[]={211, 321, 2212, 3222, 3112, 3312, 3334};
	for (int i=0;i<7;i++){
		if (charged_hadrons_pid[i]==abs(pid)){
			return true;
		}
	}
	return false;
}

  // keeps the jets which have highest pt charged particle >5GeV
bool pass(PseudoJet & jet){
	vector<PseudoJet> constituents = jet.constituents();
	for (PseudoJet& element:constituents){
		if (Check_Charged(element.user_info<MyInfo>().pid())){
			if(element.pt()>=5){
				return true;
			}   
		}
	}
	return false;
}

std::vector<int> extra_filter(std::vector<PseudoJet>& jets){
	std::vector<int> jets_index;
	for(int i=0;i<jets.size();i++){
		if(pass(jets[i])){
			jets_index.push_back(i);
		}
	}
	return jets_index; 
} 

/*
void hole_sub(vector<PseudoJet>& jets, vector<PseudoJet>& holes, double R) {
	std::vector<PseudoJet> index;
	for (int i=0;i<jets.size();i++){
		    //std::cout<<"Holes size "<<holes.size()<<std::endl;
		index.clear();
		    //std::cout<<"Index size "<<index.size()<<std::endl;
		for (int j=0;j<holes.size();j++){
			double delta_eta = jets[i].pseudorapidity() - holes[j].pseudorapidity();
			double delta_phi = jets[i].phi_std() - holes[j].phi_std();
			double delta_R = sqrt(delta_eta * delta_eta + delta_phi * delta_phi);   
			if (delta_R<=R) {
				//        std::cout<<"...YES!!...."<<i<<std::endl;
				//        std::cout<<"jet before px "<<jets[i].px()<<" py "<<jets[i].py()<<" pz "<<jets[i].pz()<<" e "<<jets[i].e()<<" pT "<<jets[i].perp()<<std::endl;
				jets[i].reset(jets[i].px() - holes[j].px(), jets[i].py() - holes[j].py(), jets[i].pz() - holes[j].pz(), jets[i].e() - holes[j].e());
				//        std::cout<<"jet after px "<<jets[i].px()<<" py "<<jets[i].py()<<" pz "<<jets[i].pz()<<" e "<<jets[i].e()<<" pT "<<jets[i].perp()<<std::endl;
				//        std::cout<<"\n";
			} 
			else{
				index.push_back(holes[j]);
			}
		}
		holes=index;
	}
}
*/

// void ProcessJet(fastjet::PseudoJet &jet, std::vector<fastjet::PseudoJet> &new_constituents){
// 	//rotate around z axis by -phi_jet
// 	double phi = std::atan2(jet.py(), jet.px());
// 	double cos_phi = std::cos(-phi);
// 	double sin_phi = std::sin(-phi);
// 	//rotate arounf y axis by -theta
// 	double theta = std::atan2(jet.pz(), jet.perp());
// 	double cos_theta = std::cos(theta);
// 	double sin_theta = std::sin(theta);

// 	auto rotate = [&](double &x, double &y, double &z) {
// 		//rotates about z
// 		double x1 = cos_phi * x - sin_phi * y;
// 		double y1 = sin_phi * x + cos_phi * y;
// 		double z1 = z;
// 		//rotates about y
// 		double x2 = cos_theta * x1 + sin_theta * z1;
// 		double y2 = y1;
// 		double z2 = -sin_theta * x1 + cos_theta * z1;
	
// 		x = x2;
// 		y = y2;
// 		z = z2;
// 	};
// 	//rotate jet constituents
// 	for (auto &c : jet.constituents()) {
// 		double cx = c.px();
// 		double cy = c.py();
// 		double cz = c.pz();
// 		rotate(cx, cy, cz);
// 		fastjet::PseudoJet new_c(cx, cy, cz, c.E());
// 		if (c.has_user_info<MyInfo>()) {
// 			const MyInfo &info = c.user_info<MyInfo>();
// 			new_c.set_user_info(new MyInfo(info)); 
// 		}
// 		new_constituents.push_back(new_c);
		
// 	}
// 	//debug
// 	//double px = jet.px();
// 	//double py = jet.py();
// 	//double pz = jet.pz();
// 	//rotate(px, py, pz);
// 	//std::cout<<"Old coordinate x:"<<jet.px()<<" y "<<jet.py()<<" z "<<jet.pz()<<std::endl;
// 	//std::cout<<"Updated coordinate x:"<<px<<" y "<<py<<" z "<<pz<<" |p| "<<std::pow(std::pow(jet.px(),2.0)+std::pow(jet.py(),2.0)+std::pow(jet.pz(),2.0),0.5)<<" E "<<jet.E()<<std::endl;	
// }
void ProcessJet(fastjet::PseudoJet &jet, std::vector<fastjet::PseudoJet> &new_constituents,std::vector<fastjet::PseudoJet> &old_constituents){
	//rotate around z axis by -phi_jet
	double phi = std::atan2(jet.py(), jet.px());
	double cos_phi = std::cos(-phi);
	double sin_phi = std::sin(-phi);
	//rotate arounf y axis by -theta
	double theta = std::atan2(jet.pz(), jet.perp());
	double cos_theta = std::cos(theta);
	double sin_theta = std::sin(theta);

	auto rotate = [&](double &x, double &y, double &z) {
		//rotates about z
		double x1 = cos_phi * x - sin_phi * y;
		double y1 = sin_phi * x + cos_phi * y;
		double z1 = z;
		//rotates about y
		double x2 = cos_theta * x1 + sin_theta * z1;
		double y2 = y1;
		double z2 = -sin_theta * x1 + cos_theta * z1;
	
		x = x2;
		y = y2;
		z = z2;
	};
	//rotate jet constituents
	for (auto &c : jet.constituents()) {
		double cx = c.px();
		double cy = c.py();
		double cz = c.pz();
		rotate(cx, cy, cz);
		fastjet::PseudoJet new_c(cx, cy, cz, c.E());
		fastjet::PseudoJet old_c(c.px(), c.py(), c.pz(), c.E());
		if (c.has_user_info<MyInfo>()) {
			const MyInfo &info = c.user_info<MyInfo>();
			new_c.set_user_info(new MyInfo(info)); 
			old_c.set_user_info(new MyInfo(info));
		}
		new_constituents.push_back(new_c);
		old_constituents.push_back(old_c);
	}
	//debug
	// double px = jet.px();
	// double py = jet.py();
	// double pz = jet.pz();
	// rotate(px, py, pz);
	// std::cout<<"Old coordinate x:"<<jet.px()<<" y "<<jet.py()<<" z "<<jet.pz()<<std::endl;
	// std::cout<<"Updated coordinate x:"<<px<<" y "<<py<<" z "<<pz<<" |p| "<<std::pow(std::pow(jet.px(),2.0)+std::pow(jet.py(),2.0)+std::pow(jet.pz(),2.0),0.5)<<" E "<<jet.E()<<std::endl;	
}


//------------------------------------------------------------
/// MAIN PROGRAM
// int FastJetProcedure(int lower_lim,int upper_lim,int number){
//   //input file name  
//   std::string input_file="/rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0/jetmod_" + std::to_string(number) + "_final_state_hadrons.dat";
//   //output file name
//   std::string output_file="/rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0/jetmod_" + std::to_string(number) + "_final_state_hadrons_processed.dat";
int FastJetProcedure(int lower_lim,int upper_lim,int number,
                     const std::string& input_override="",
                     const std::string& output_override="") {
	std::string base_simulation_path = "/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/";						
	std::string input_file  = input_override.empty()  
								? base_simulation_path+(std::string("c0005_alpha0p20_q_1p0/jetmod_") + std::to_string(number) + "_final_state_hadrons.dat") 
								: input_override;
	std::string output_file = output_override.empty() 
  							? base_simulation_path+(std::string("c0005_alpha0p20_q_1p0_unfolded/jetmod_") + std::to_string(number) + "_final_state_hadrons.dat")
				            : output_override;

  double sigma, sigmaerr;
  std::ifstream file0(input_file);  
  std::ofstream file1;
  if (file0.is_open()){
    std::cout<<"Input File successfully opened: "<<input_file<<std::endl;
    file1.open(output_file);
	std::cout<<"Output File successfully opened: "<<output_file<<std::endl;
    file1<< std::setw(5) << "index" << std::setw(15) << "particle_id"<< std::setw(15) << "E(GeV)" << std::setw(15) << "pT(GeV)" << std::setw(15)<< "phi" << std::setw(15)<< "eta"<<std::endl;
  }
  else{
    std::cout<<"Failed to open the file "<<input_file<<std::endl;
    return 0;
  }

  int i,j,event;
  event=0;
  std::string str;
  float values[9];		// index >>PID>>particle_status>> E >> px >> py >> pz >> eta>> phi

  vector<fastjet::PseudoJet> input_particles;
  vector<fastjet::PseudoJet> hole_particles;  
  double R = 0.4; //radius jet
  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
  double weight, weightb;
  double EPlane;
  char hash;

  std::string s;
  while(std::getline(file0,str)){                 //std::getline(file0,str)){
    i=0;
    j=0;
   // std::cout<<str[0]<<" "<<str[1]<<" "<<str[2]<<" "<<str[3]<<" "<<str[4]<<" "<<str[5]<<" "<<str[6]<<" "<<str[7];
//    if (str[0]=='#' && str[4]=='E'){
       //std::cout<<str[0]<<" "<<str[2]<<" "<<str[5]<<std::endl;
       if (str[0]=='#' && str[2]=='E'){
                        std::istringstream iss(str);
			std::string token;
			for(int i=0; i<7; i++){
				iss >> token;
				if (i == 4){
					weight = std::stod(token);
				}
                                if (i == 6){
                                         EPlane = std::stod(token);
				}
			} 
			if (event==0){weightb = weight;}      
			if(event>0){        //still there is something to be done about the last event 4000
				//std::cout<<"New Event "<<event<<std::endl;
				//file1<<"Event "<<event<< std::fixed << std::setprecision(10)<<std::setw(15)<< weightb <<std::setw(15)<<EPlane<<"\n";
				fastjet::ClusterSequence clust_seq(input_particles,jet_def);
				vector<PseudoJet> inclusive_jets=clust_seq.inclusive_jets();
				//vector<int> jet_index=extra_filter(inclusive_jets);
				//hole_sub(inclusive_jets,hole_particles,R);
				for (auto& jet : inclusive_jets){
					std::vector<fastjet::PseudoJet> new_constituents;
					std::vector<fastjet::PseudoJet> old_constituents;
					if (jet.perp() >= 50.0 &&  jet.perp() <= 250.0 && std::abs(jet.rap())<=2.8){
						//file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
						ProcessJet(jet, new_constituents, old_constituents);
						for (int i=0;i<new_constituents.size();i++) {
							file1<< std::setw(5) << i << std::setw(15) << new_constituents[i].user_info<MyInfo>().pid() << std::setw(15) << new_constituents[i].E() << std::setw(15) << new_constituents[i].perp() << std::setw(15) <<old_constituents[i].phi() << std::setw(15) << old_constituents[i].pseudorapidity() << std::endl;
						}
						file1<< std::setw(5) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::endl;
					}
				}
				/*
				file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
				for (int i=0;i<inclusive_jets.size();i++) {
					file1<< std::setw(5) << i << std::setw(15) << inclusive_jets[i].rap()<< std::setw(15) << inclusive_jets[i].phi() << std::setw(15) << inclusive_jets[i].perp() << std::setw(15) << inclusive_jets[i].E()<< std::endl;
				}*/
				//file1<<"\n";
				weightb = weight;
			}
      event+=1;
      input_particles.clear();
      hole_particles.clear();
      continue;
    }
    if (str[0]=='#' && str[2]=='J'){continue;}
    if (str[0]=='#' && str[2]=='s'){
       std::istringstream iss(str);
       std::string token;
       for(int i=0; i<5; i++){
	iss >> token;
	if (i == 2){
	    sigma = std::stod(token);
	}
        if (i == 4){sigmaerr = std::stod(token);}
	} 
    }   
    while (i!=str.size()){//(str[i]!='\0'){
      if(str[i]==' '||str[i]=='\t'||i==str.size()-1){
        values[j]=std::atof(s.c_str());
        s.clear();
        j+=1;
      }   
      else{
        s+=str[i];
      }   
      i+=1;
    }
    if (values[2]<0){
      //std::cout<<"Some one was here "<<std::endl;
      PseudoJet p(values[4],values[5],values[6],values[3]); //px,py,pz,
      p.set_user_info(new MyInfo(values[2],values[1]));   
      hole_particles.push_back(p);
    }  
    else{
      //std::cout<<"Status id is "<<values[2]<<std::endl;
      if (values[2]>=0 && values[2]<2){
        PseudoJet p(values[4],values[5],values[6],values[3]); //px,py,pz,
        p.set_user_info(new MyInfo(values[2],values[1]));   
        input_particles.push_back(p); 
      }   
    } 
  }   
  	std::cout<<"New Event "<<event<<std::endl;  
	//file1<<"Event "<<event<< std::fixed << std::setprecision(10)<<std::setw(15)<< weightb <<std::setw(15)<<EPlane<<"\n";
	fastjet::ClusterSequence clust_seq(input_particles,jet_def);
	vector<PseudoJet> inclusive_jets=clust_seq.inclusive_jets();

	//------------------------------hole sub------------------------------------------------//
	//--------------------------------------------------------------------------------------//

	//std::vector<int> jet_index=extra_filter(inclusive_jets);
	//hole_sub(inclusive_jets,hole_particles,R);
	for (auto& jet : inclusive_jets){
		std::vector<fastjet::PseudoJet> new_constituents;
		std::vector<fastjet::PseudoJet> old_constituents;
		if (jet.perp() >= 50.0 &&  jet.perp() <= 250.0 && std::abs(jet.rap())<=2.8){
			//file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
			ProcessJet(jet, new_constituents, old_constituents);
			for (int i=0;i<new_constituents.size();i++) {
				//const MyInfo& info = new_constituents[i].user_info<MyInfo>();
				file1<< std::setw(5) << i << std::setw(15) << new_constituents[i].user_info<MyInfo>().pid() << std::setw(15) << new_constituents[i].E() << std::setw(15) << new_constituents[i].perp() << std::setw(15) <<old_constituents[i].phi() << std::setw(15) << old_constituents[i].pseudorapidity() << std::endl;
			}
			file1<< std::setw(5) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::setw(15) << 0 << std::endl;
		}
	}


	/*
	file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
	for (int i=0;i<inclusive_jets.size();i++) {
		
		file1<< std::setw(5) << i << std::setw(15) << inclusive_jets[i].rap()<< std::setw(15) << inclusive_jets[i].phi() << std::setw(15) << inclusive_jets[i].perp() << std::setw(15) << inclusive_jets[i].E() << std::endl;
	}
 	//file1<< std::setw(5) << "sigma" <<std::fixed << std::setprecision(10)<< std::setw(15) << sigma<< std::setw(15) << "sigmaerr" << std::setw(15) << sigmaerr<< std::endl;
        file1 << std::scientific << std::setprecision(6)<< "sigma " << sigma << " sigmaerr " << sigmaerr << "\n";
	*/
	file0.close();  
	file1.close();
	// read in input particles
	//----------------------------------------------------------
	return 0;
}

int main(int argc, char* argv[]){
    int low_lim=0;//atoi(argv[1]);
    int upper_lim=0;//atoi(argv[2]);
    // int number = atoi(argv[1]);
    // FastJetProcedure(low_lim,upper_lim,number);
	if (argc == 2) {
		cout << "Single argument mode\n"<< "  Argument: " << argv[1] << "\n";

       int number = atoi(argv[1]);
       return FastJetProcedure(low_lim, upper_lim, number);
	} else if (argc == 3) {
		// cout << "Two argument mode\n"<< "  Input file: " << argv[1] << "\n"<< "  Output file: " << argv[2] << "\n";

		// return FastJetProcedure(low_lim, upper_lim, /*number=*/0, argv[1], argv[2]);
		  	std::string in_dir = argv[1];
			std::string out_dir = argv[2];

			if (!is_dir(in_dir)) {
				std::cerr << "ERROR: input_dir not found or not a directory: " << in_dir << "\n";
				return 1;
			}
			ensure_dir(out_dir);

			// Collect *.dat that match our expected naming
			std::vector<std::string> files;
			DIR* d = ::opendir(in_dir.c_str());
			if (!d) {
				std::cerr << "ERROR: cannot open: " << in_dir << "\n";
				return 1;
			}
			for (dirent* e = ::readdir(d); e; e = ::readdir(d)) {
				if (e->d_type == DT_REG || e->d_type == DT_UNKNOWN) {
				std::string name = e->d_name;
				// simplest filter: require both substrings and .dat suffix
				if (name.find("jetmod_") == 0 &&
					name.find("_final_state_hadrons.dat") != std::string::npos) {
					files.push_back(in_dir + "/" + name);
				}
				}
			}
			::closedir(d);
			std::sort(files.begin(), files.end()); // jetmod_0..199 in order

			// Report how many we found (expect 200)
			std::cout << "[jet_unfolding] Found " << files.size()
						<< " input .dat files in " << in_dir << " (expected 200)\n";

			int ok = 0, fail = 0;
			for (const auto& in_path : files) {
				std::string out_path = out_dir + "/" + basename_of(in_path); // same filename
				cout << "Processing file: " << in_path << " -> " << out_path << "\n";
				int rc = FastJetProcedure(low_lim, upper_lim, /*number=*/0, in_path, out_path);
				if (rc == 0) ++ok; else ++fail;
			}

			std::cout << "[jet_unfolding] Done. processed=" << ok << " failed=" << fail << "\n";
			return (fail == 0) ? 0 : 2;
	} else {
		std::cerr << "Usage: " << argv[0] << " <number>\n"
					<< "   or: " << argv[0] << " <input_full_path> <output_full_path>\n";
		return 1;
	}
    return 0;
}

// g++ -std=c++11 jet_unfolding.cc -o jet_unfolding $(/wsu/el7/gnu7/fastjet/3.4.0/bin/fastjet-config --cxxflags --libs --plugins)"

/* usage:
./jet_unfolding 0

or
./jet_unfolding /rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0/jetmod_0_final_state_hadrons.dat /rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0_unfolded/jetmod_0_final_state_hadrons.dat
*/
