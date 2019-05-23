// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <stack>

using namespace std;
using namespace cv;

class graph_utils
{
	public:
	Mat Gadj; //adjacency matrix
	vector<bool> comp_set; //if already has a component associated -- all true at the end
	vector<vector<int> > c_components; //each position will have an entire connected component set (composed by the vertice idxs)


graph_utils(Mat G);

Mat getG();
void setG(Mat G);

vector<int> get_cc_elements(int idx); // get the connected component elements given an arbitrary vertice as start
vector<vector<int> > get_all_cc(); //get all the connected components in the graph -> it fills the c_components vector
vector<int> get_inv_cc(vector<int> cc); //returnn the vertices that are not in the input cc
};



