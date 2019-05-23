// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



#include "graph_utils.h"


graph_utils::graph_utils(Mat G)
{
	Gadj = G.clone();
	comp_set.resize(G.rows);

	for(int i=0;i< comp_set.size();i++)
		comp_set[i] = false;
}

Mat graph_utils::getG()
{
	return Gadj;
}

void graph_utils::setG(Mat G)
{
	Gadj = G.clone();
}

vector<int> graph_utils::get_cc_elements(int idx)
{
	stack<int> recursion;
	vector<int> cc;
	int curr;
	recursion.push(idx);
	comp_set[idx] = true;

	while(!recursion.empty())
	{
		curr = recursion.top(); recursion.pop();
		cc.push_back(curr);

		//////////pushing child vertices

		for(int i=0;i<Gadj.rows;i++)
		{
			if(Gadj.at<double>(curr,i) != 0 && !comp_set[i])
			{
				comp_set[i] = true;
				recursion.push(i);	
			}
			else if(Gadj.at<double>(i,curr) != 0 && !comp_set[i])
			{
				comp_set[i] = true;
				recursion.push(i);
			}
		}

	}

	return cc;
}

vector<vector<int > > graph_utils::get_all_cc()
{
	vector<vector<int> > comps;

	for(int i=0;i<comp_set.size();i++)
	{
		if(!comp_set[i])
			comps.push_back(get_cc_elements(i));
	}

	c_components = comps;
	return comps;
}

vector<int> graph_utils::get_inv_cc(vector<int> cc)
{
	vector<int> inv_cc;
	vector<bool> all_v(comp_set.size(),true);


	for(int i=0;i<cc.size();i++)
		all_v[cc[i]] = false;

	for(int i=0;i<all_v.size();i++)
		if(all_v[i])
			inv_cc.push_back(i);

	return inv_cc;
	
}
