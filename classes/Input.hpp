#include <fstream>
#include <vector>

class Input {
	private:
		std::ifstream inFile;
		int currentData = 0;
		int maxData;
	
	public:
		int inputShape, outputShape;
		Input(){
			this->Init();
		}
		~Input(){
			this->inFile.close();
		}
		auto Init() -> void;
		auto NextData() -> std::vector<float>;
		auto NextTarget() -> std::vector<float>;
};
