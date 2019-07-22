#include <time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

static const int num_threads = 2;

namespace paddle {
namespace train {

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<paddle::framework::ProgramDesc> Load(
    paddle::framework::Executor* executor, const std::string& model_filename) {
  VLOG(3) << "loading model from " << model_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<paddle::framework::ProgramDesc> main_program(
      new paddle::framework::ProgramDesc(program_desc_str));
  return main_program;
}

}  // namespace train
}  // namespace paddle

//load mnist dataset
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
 
void read_Mnist_Label(std::string filename, std::vector<int64_t>& labels)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		std::cout << "magic number = " << magic_number << std::endl;
		std::cout << "number of images = " << number_of_images << std::endl;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((int64_t)label);
		}
	}
}
 
void read_Mnist_Images(std::string filename, std::vector<std::vector<float>>&images)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
 
		std::cout << "magic number = " << magic_number << std::endl;
		std::cout << "number of images = " << number_of_images << std::endl;
		std::cout << "rows = " << n_rows << std::endl;
		std::cout << "cols = " << n_cols << std::endl;
 
		for (int i = 0; i < number_of_images; i++)
		{
			std::vector<float> tp;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					tp.push_back((float)image);
				}
			}
			images.push_back(tp);
		}
	}
}

//train model thread
void ThreadedRunTrain(
    const std::unique_ptr<paddle::framework::ProgramDesc>& train_program,
    paddle::framework::Executor* executor, 
    paddle::framework::Scope* scope,
    std::vector<std::vector<float>>& images,
    std::vector<int64_t>& labels,
    const std::string& loss_name,
    const int thread_id) {

  auto& copy_scope = scope->NewScope();
  auto loss_var = copy_scope.Var(loss_name);
  //load data
  //X
  int batch_size = 128;
  int image_size = 28 * 28;
  auto x_var = copy_scope.Var("image");
  auto x_tensor = x_var->GetMutable<paddle::framework::LoDTensor>();
  x_tensor->Resize({batch_size, 1, 28, 28});
  auto x_data = x_tensor->mutable_data<float>(paddle::platform::CPUPlace());
  //Y
  auto y_var = copy_scope.Var("label");
  auto y_tensor = y_var->GetMutable<paddle::framework::LoDTensor>();
  y_tensor->Resize({batch_size, 1});
  auto y_data = y_tensor->mutable_data<int64_t>(paddle::platform::CPUPlace());
  //set X Y value
  for(int epoch = thread_id; epoch < 100 * num_threads;epoch += num_threads){
    for(int i = 0;i < batch_size; ++i){
        int index = (i + epoch * batch_size) % labels.size();
        for(int j = 0;j < image_size; ++j){
                 x_data[i * image_size + j] = static_cast<float>(images[index][j] / 255.0);
      }
      y_data[i] = static_cast<int64_t>(labels[index]);
   }
   executor->Run(*train_program, &copy_scope, 0, false, true);
   /* for(auto& op_desc : train_program->Block(0).AllOps()){
        auto op =paddle::framework:: OpRegistry::CreateOp(*op_desc);
           op->Run(copy_scope,paddle::platform::CPUPlace());
      }*/
    std::cout << "thread_" << thread_id << "   train loss   :"
              << loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0] 
              << std::endl;
  }
}

//save model and params
void SaveModel(const std::string &dir,
              std::unique_ptr<paddle::framework::ProgramDesc>& train_program,
              paddle::framework::Scope* scope
              ){
  //save model
  std::string model_name = dir + "/model";
  std::ofstream outfile;
  outfile.open(model_name, std::ios::out | std::ios::binary);
  std::string prog_desc = train_program->Proto()->SerializeAsString();;
  outfile << prog_desc;
 
 // save params 
  paddle::framework::ProgramDesc save_program;
  auto *save_block = save_program.MutableBlock(0);
  const paddle::framework::ProgramDesc &main_program = *train_program;
  const paddle::framework::BlockDesc &global_block = main_program.Block(0);
  std::vector<std::string> save_var_list;
  for (paddle::framework::VarDesc *var : global_block.AllVars()) {
    if (var->Persistable()) {
      paddle::framework::VarDesc *new_var = save_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      save_var_list.push_back(new_var->Name());
     }
   std::cout << var->Name() << std::endl;
  }

 // paddle::platform::CPUPlace place;
 // paddle::framework::Executor exe(place);
 // exe.Run(save_program, scope, 0);

  std::sort(save_var_list.begin(), save_var_list.end());
  auto *op = save_block->AppendOp();
  op->SetType("save_combine");
  op->SetInput("X", save_var_list);
  op->SetAttr("file_path", dir + "/params");
  op->CheckAttrs();
  paddle::platform::CPUPlace place;
  paddle::framework::Executor exe(place);
  
  auto op2 =paddle::framework:: OpRegistry::CreateOp(*op);
  op2->Run(*scope,paddle::platform::CPUPlace());
}

int main() {
  paddle::framework::InitDevices(false);
  const auto cpu_place = paddle::platform::CPUPlace();
  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program = paddle::train::Load(&executor, "startup_program");
  auto train_program = paddle::train::Load(&executor, "main_program");

  std::string loss_name = "";
  for (auto op_desc : train_program->Block(0).AllOps()) {
    if (op_desc->Type() == "mean") {
      loss_name = op_desc->Output("Out")[0];
      break;
    }
  }
  PADDLE_ENFORCE_NE(loss_name, "", "loss not found");
  // init all parameters
  executor.Run(*startup_program, &scope, 0);
  auto loss_var = scope.Var(loss_name);

  //load mnist
  std::vector<std::vector<float> > images;
  std::vector<int64_t> labels;
  read_Mnist_Images("train-images-idx3-ubyte", images);
  read_Mnist_Label("train-labels-idx1-ubyte",labels);
 
 // multi processing
  std::vector<std::thread*> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.push_back(new std::thread(
        ThreadedRunTrain, std::ref(train_program), &executor, &scope,std::ref(images),std::ref(labels),loss_name,i));
  }

  for (int i = 0; i < num_threads; ++i) {
    threads[i]->join();
    delete threads[i];
  }
 // save model 
  SaveModel("breakpoint", train_program,&scope);
}
