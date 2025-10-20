#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include <iostream> // needed for io
#include <cstdio>   // needed for io
#include <fstream>
#include <cmath>
#include <iomanip> // For setw function
#include <string>

using namespace std;
using namespace fastjet;

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

void ProcessJet(fastjet::PseudoJet &jet, std::vector<fastjet::PseudoJet> &new_constituents){
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
		new_constituents.emplace_back(cx, cy, cz, c.E());
		
	}
	//debug
	//double px = jet.px();
	//double py = jet.py();
	//double pz = jet.pz();
	//rotate(px, py, pz);
	//std::cout<<"Old coordinate x:"<<jet.px()<<" y "<<jet.py()<<" z "<<jet.pz()<<std::endl;
	//std::cout<<"Updated coordinate x:"<<px<<" y "<<py<<" z "<<pz<<" |p| "<<std::pow(std::pow(jet.px(),2.0)+std::pow(jet.py(),2.0)+std::pow(jet.pz(),2.0),0.5)<<" E "<<jet.E()<<std::endl;	
}

//------------------------------------------------------------
/// MAIN PROGRAM
int FastJetProcedure(int lower_lim,int upper_lim,int number){
  //input file name  
  std::string input_file="/rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0/jetmod_" + std::to_string(number) + "_final_state_hadrons.dat";
  //output file name
  std::string output_file="/rs/rs_grp_mljet/ml-jet-ext/simulation_results/c0005_alpha0p20_q_1p0/jetmod_" + std::to_string(number) + "_final_state_hadrons_processed.dat";

  double sigma, sigmaerr;
  std::ifstream file0(input_file);  
  std::ofstream file1;
  if (file0.is_open()){
    std::cout<<"File successfully opened"<<std::endl;
    file1.open(output_file);
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
				file1<<"Event "<<event<< std::fixed << std::setprecision(10)<<std::setw(15)<< weightb <<std::setw(15)<<EPlane<<"\n";
				fastjet::ClusterSequence clust_seq(input_particles,jet_def);
				vector<PseudoJet> inclusive_jets=clust_seq.inclusive_jets();
				//vector<int> jet_index=extra_filter(inclusive_jets);
				//hole_sub(inclusive_jets,hole_particles,R);
				for (auto& jet : inclusive_jets){
					std::vector<fastjet::PseudoJet> new_constituents;
						if (jet.perp() >= 100.0 &&  jet.perp() <= 110.0 && std::abs(jet.rap())<=2.0){
						file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
						ProcessJet(jet, new_constituents);
						for (int i=0;i<new_constituents.size();i++) {
							file1<< std::setw(5) << i << std::setw(15) << new_constituents[i].rap()<< std::setw(15) << new_constituents[i].phi() << std::setw(15) << new_constituents[i].perp() << std::setw(15) << new_constituents[i].E() << std::endl;
						}
					}
				}
				/*
				file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
				for (int i=0;i<inclusive_jets.size();i++) {
					file1<< std::setw(5) << i << std::setw(15) << inclusive_jets[i].rap()<< std::setw(15) << inclusive_jets[i].phi() << std::setw(15) << inclusive_jets[i].perp() << std::setw(15) << inclusive_jets[i].E()<< std::endl;
				}*/
				file1<<"\n";
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
	file1<<"Event "<<event<< std::fixed << std::setprecision(10)<<std::setw(15)<< weightb <<std::setw(15)<<EPlane<<"\n";
	fastjet::ClusterSequence clust_seq(input_particles,jet_def);
	vector<PseudoJet> inclusive_jets=clust_seq.inclusive_jets();

	//------------------------------hole sub------------------------------------------------//
	//--------------------------------------------------------------------------------------//

	//std::vector<int> jet_index=extra_filter(inclusive_jets);
	//hole_sub(inclusive_jets,hole_particles,R);
	for (auto& jet : inclusive_jets){
		std::vector<fastjet::PseudoJet> new_constituents;
		if (jet.perp() >= 100.0 &&  jet.perp() <= 110.0 && std::abs(jet.rap())<=2.0){
			file1<< std::setw(5) << "jet #" << std::setw(15) << "rapidity"<< std::setw(15) << "phi" << std::setw(15) << "pt" << std::setw(15)<< "E" <<std::endl;
			ProcessJet(jet, new_constituents);
			for (int i=0;i<new_constituents.size();i++) {
				file1<< std::setw(5) << i << std::setw(15) << new_constituents[i].rap()<< std::setw(15) << new_constituents[i].phi() << std::setw(15) << new_constituents[i].perp() << std::setw(15) << new_constituents[i].E() << std::endl;
			}
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
    int number = atoi(argv[1]);
    FastJetProcedure(low_lim,upper_lim,number);
    return 0;
}
