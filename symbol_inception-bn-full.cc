/* ----------------------------------------------------------------------------
*  Caffe prototxt generator for Inception21k. Code adapted from the original model definition at
*  https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol_inception-bn.py
*  Author: Antonio Pertusa (pertusa AT ua DOT es)
*  License: GNU Public License
* ----------------------------------------------------------------------------*/

#include <iostream>
using namespace std;

string printVariable(string name)
{
    cout << "layer { " << endl <<
            "  name: \"" << name << "\"" << endl <<
            "  type: \"Input\"" << endl <<
            "  top: \"" << name << "\"" << endl <<
            "  input_param { shape: { dim: 10 dim: 3 dim: 224 dim: 224 } }" << endl << 
            "}" << endl << endl;
    return name;
}

string printConvolution(string prevLayerName, int num_filter, int kernel, int stride, int pad, string name)
{
  cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Convolution\"" << endl <<
          "  bottom: \"" << prevLayerName << "\"" << endl <<
          "  top: \"" << name << "\"" << endl <<
          "  convolution_param { " << endl <<
          "     num_output: " << num_filter << endl <<
          "     kernel_size: " << kernel << endl <<
          "     stride: " << stride << endl <<
          "     pad: " << pad << endl <<
          "  }" << endl <<
          "}" << endl << endl;
  return name;
}

string printPooling(string prevLayerName, int kernel, int stride, int pad, string pool_type, string name)
{
   cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Pooling\"" << endl <<
          "  bottom: \"" << prevLayerName << "\"" << endl <<
          "  top: \"" << name << "\"" << endl <<
          "  pooling_param {" << endl <<
          "     pool: " << pool_type << endl <<
          "     kernel_size: " << kernel << endl <<
          "     stride: " << stride << endl <<
          "     pad: " << pad << endl <<
          "  }" << endl <<
          "}" << endl << endl;

    return name;
}

string printGlobalPooling(string prevLayerName, int kernel, int stride, int pad, string name, string pool_type)
{
    cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Pooling\"" << endl <<
          "  bottom: \"" << prevLayerName << "\"" << endl <<
          "  top: \"" << name << "\"" << endl <<
          "  pooling_param {" << endl <<
          "     global_pooling : true" << endl << 
          "     pool: " << pool_type << endl <<
          "  }" << endl <<
          "}" << endl << endl;

    return name;
}


string printActivation(string prevLayerName, string act_type, string name)
{
  cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"" << act_type << "\"" << endl <<
          "  bottom: \"" << prevLayerName << "\"" << endl <<
          "  top: \"" << prevLayerName << "\"" << endl <<
          "}" << endl << endl ;

  return name;
}

string printConcat4(const string &prevLayerName1, const string &prevLayerName2, const string &prevLayerName3, const string &prevLayerName4, const string &name)
{
  cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Concat\"" << endl <<
          "  bottom: \"" << prevLayerName1 << "\"" << endl <<
          "  bottom: \"" << prevLayerName2 << "\"" << endl <<
          "  bottom: \"" << prevLayerName3 << "\"" << endl <<
          "  bottom: \"" << prevLayerName4 << "\"" << endl <<
          "  top: \""  << name << "\"" << endl <<
          "}" << endl << endl;

  return name;
}

string printConcat3(string prevLayerName1, string prevLayerName2, string prevLayerName3, string name)
{
  cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Concat\"" << endl <<
          "  bottom: \"" << prevLayerName1 << "\"" << endl <<
          "  bottom: \"" << prevLayerName2 << "\"" << endl <<
          "  bottom: \"" << prevLayerName3 << "\"" << endl <<
          "  top: \""  << name << "\"" << endl <<
          "}" << endl << endl;

  return name;
}


string printSoftmaxOutput(string prevLayerName, string name)
{
  cout << "layer {" << endl <<
          "  name: \"" << name << "\"" << endl <<
          "  type: \"Softmax\"" << endl <<
          "  bottom: \"" << prevLayerName << "\"" << endl <<
          "  top: \"" << name << "\"" << endl <<
          "}" << endl << endl;

  return name;
}

string printFullyConnected(string prevLayerName, int num_output, string name)
{
  cout << "layer {" << endl <<
    "  name: \"" << name << "\"" << endl <<
    "  type: \"InnerProduct\"" << endl <<
    "  bottom: \"" << prevLayerName << "\"" << endl <<
    "  top: \"" << name << "\"" << endl <<
    "  param {" << endl <<
    "    lr_mult: 1" << endl <<
    "    decay_mult: 1" << endl <<
    "  }" << endl <<
    "  param {" << endl <<
    "    lr_mult: 2" << endl <<
    "    decay_mult: 0" << endl <<
    "  }" << endl <<
    "  inner_product_param {" << endl <<
    "    num_output: " << num_output << endl <<
    "    weight_filler {" << endl <<
    "      type: \"xavier\"" << endl <<
    "    }" << endl <<
    "    bias_filler {" << endl <<
    "      type: \"constant\"" << endl <<
    "      value: 0" << endl <<
    "    }" << endl <<
    "  }" << endl <<
    "}" << endl << endl;

    return name;
}

string printBatchNorm(string prevLayerName, string name)
{
  cout << "layer {" << endl <<
    "  name: \"" << name << "\"" << endl <<
    "  type: \"BatchNorm\"" << endl <<
    "  bottom: \"" << prevLayerName << "\"" << endl <<
    "  top: \"" << prevLayerName << "\"" << endl <<
    "  batch_norm_param {" << endl <<
    "    use_global_stats: true" << endl << // false for training, true for test: https://github.com/BVLC/caffe/issues/3347
    "  }" << endl <<
 //   "  param {" << endl <<
 //   "    lr_mult: 0" << endl <<
 //   "  }" << endl <<
 //   "  param {" << endl <<
 //   "    lr_mult: 0" << endl <<
 //   "  }" << endl <<
 //   "  param {" << endl <<
 //   "    lr_mult: 0" << endl <<
 //   "  }" << endl <<
    "}" << endl << endl <<
    "layer {" << endl << 
    "  name: \"scale_" << prevLayerName << "\"" << endl << 
    "  bottom: \"" << prevLayerName << "\"" << endl <<
    "  top: \"" << prevLayerName << "\"" << endl <<
    "  type: \"Scale\"" << endl <<
    "  scale_param {" << endl <<
    "        bias_term: true" << endl << 
    "  }" << endl << 
    "}" << endl << endl;

    return name;
}

//////////// Utils ///////////

string concat(const string &s1, const string &s2, const string &s3="", const string &s4="", const string &s5="")
{
  string output=s1+s2+s3+s4+s5;
  return output;
}

string tolower(const string &s)
{
  string output;
  for(int i = 0; i<s.length(); i++)
      output += tolower(s[i]);
  return output;
}

//////////// Factory methods ///////////

string ConvFactory(string data, int num_filter, int kernel, int stride, int pad, string name="", string suffix="")
{
    string conv = printConvolution(data, num_filter, kernel, stride, pad, concat("conv_",name, suffix));
    string bn = printBatchNorm(conv, concat("bn_",name,suffix));
    string act = printActivation(conv, "ReLU", concat("relu_",name, suffix));
    return conv;
}

string InceptionFactoryA(string data, int num_1x1, int num_3x3red, int num_3x3, int num_d3x3red, int num_d3x3, string pool, int proj, string name)
{
    // 1x1
    string c1x1 = ConvFactory(data, num_1x1, 1, 1, 0, concat(name,"_1x1"));
    // 3x3 reduce + 3x3
    string c3x3r = ConvFactory(data, num_3x3red, 1, 1, 0, concat(name,"_3x3"),"_reduce");
    string c3x3 = ConvFactory(c3x3r, num_3x3, 3, 1, 1, concat(name,"_3x3"));
    // double 3x3 reduce + double 3x3
    string cd3x3r = ConvFactory(data, num_d3x3red, 1, 1, 0, concat(name,"_double_3x3"), "_reduce");
    string cd3x3 = ConvFactory(cd3x3r, num_d3x3, 3, 1, 1, concat(name,"_double_3x3_0"));
    cd3x3 = ConvFactory(cd3x3, num_d3x3, 3, 1, 1, concat(name,"_double_3x3_1"));
    // pool + proj
    string pooling = printPooling(data, 3, 1, 1, pool, concat(tolower(pool),"_pool_",name,"_pool"));
    string cproj = ConvFactory(pooling, proj, 1, 1, 0, concat(name,"_proj"));
    // concat
    string concat_ = printConcat4(c1x1, c3x3, cd3x3, cproj, concat("ch_concat_",name,"_chconcat"));
    return concat_;
}

string InceptionFactoryB(string data, int num_3x3red, int num_3x3, int num_d3x3red, int num_d3x3, string name)
{
    // 3x3 reduce + 3x3
    string c3x3r = ConvFactory(data, num_3x3red, 1, 1, 0, concat(name,"_3x3"), "_reduce");
    string c3x3 = ConvFactory(c3x3r, num_3x3, 3, 2, 1, concat(name,"_3x3"));
    // double 3x3 reduce + double 3x3
    string cd3x3r = ConvFactory(data, num_d3x3red, 1, 1, 0, concat(name,"_double_3x3"), "_reduce");
    string cd3x3 = ConvFactory(cd3x3r, num_d3x3, 3, 1, 1, concat(name,"_double_3x3_0"));
    cd3x3 = ConvFactory(cd3x3, num_d3x3, 3, 2, 1, concat(name,"_double_3x3_1"));
    // pool + proj
    string pooling = printPooling(data, 3, 2, 0, "MAX", concat("max_pool_",name,"_pool"));
    // concat
    string concat_ = printConcat3(c3x3, cd3x3, pooling, concat("ch_concat_",name,"_chconcat"));
    return concat_;
}


int main()
{
    // data
    cout << "name: \"Inception21k\"" << endl;
    string data = printVariable("data");
    // stage 1
    string conv1 = ConvFactory(data, 96, 7, 2, 3, "conv1");
    

    string pool1 = printPooling(conv1, 3, 2, 0, "MAX", "pool1"); // Pooling is done as in previous MXNet version (conv are round up, pool are round down): https://github.com/dmlc/mxnet/issues/2718
//    string pool1 = printPooling(conv1, 3, 2, 1, "MAX", "pool1");
    // stage 2
    string conv2red = ConvFactory(pool1, 128, 1, 1, 0, "conv2red");
    string conv2 = ConvFactory(conv2red, 288, 3, 1, 1, "conv2");
    string pool2 = printPooling(conv2, 3, 2, 0, "MAX", "pool2");  // Pooling is done as in previous MXNet version (conv are round up, pool are round down)
//    string pool2 = printPooling(conv2, 3, 2, 1, "MAX", "pool2");
    // stage 2
    string in3a = InceptionFactoryA(pool2, 96, 96, 96, 96, 144, "AVE", 48, "3a");
    string in3b = InceptionFactoryA(in3a, 96, 96, 144, 96, 144, "AVE", 96, "3b");
    string in3c = InceptionFactoryB(in3b, 192, 240, 96, 144, "3c");
    // stage 3
    string in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "AVE", 128, "4a");
    string in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "AVE", 128, "4b");
    string in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "AVE", 128, "4c");
    string in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 96, "AVE", 128, "4d");
    string in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e");
    // stage 4
    string in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "AVE", 128, "5a");
    string in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "MAX", 128, "5b");
    // global avg pooling
    string avg = printGlobalPooling(in5b, 7, 1, 0, "global_pool", "AVE");
    // linear classifier
//    string flatten = printFlatten(avg, "flatten"); // Unnecessary with global_pool
    string fc1 = printFullyConnected(avg, 21841, "fc1");
    string softmax = printSoftmaxOutput(fc1, "softmax");
}
