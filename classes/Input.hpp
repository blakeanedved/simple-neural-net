#include <fstream>
#include <vector>

class Input {
	private:
		std::ifstream dataTrainFile;
		std::ifstream labelTrainFile;
		std::ifstream dataTestFile;
		std::ifstream labelTestFile;
	
	public:
		int inputShape, outputShape;
		Input(){
			this->Init();
		}
		~Input(){
			this->dataTrainFile.close();
			this->labelTrainFile.close();
			this->dataTestFile.close();
			this->labelTestFile.close();
		}
		auto Init() -> void;
		auto NextTrainData() -> std::vector<float>;
		auto NextTrainTarget() -> std::vector<float>;
		auto NextTestData() -> std::vector<float>;
		auto NextTestTarget() -> std::vector<float>;
};
