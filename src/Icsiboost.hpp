//
//   ------------------------------------------------------------------
//      Sfann v0.1 : Simple and Fast Artificial Neural Networks
//   ------------------------------------------------------------------
//
//      Copyright (C) 2010 Stanislas Oger
//
//   ..................................................................
//
//      This file is part of Sfann
//
//      Sfann is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
//   ..................................................................
//
//      Contact :
//                stanislas.oger@gmail.com
//   ..................................................................
//


#ifndef __LIB_ICSIBOOST__
#define __LIB_ICSIBOOST__

#include <sstream>
#include <fstream>
#include <string>
// #include <iterator>
// #include <algorithm>
#include <vector>
#include <map>
#include "fann.h"
#include "SfannException.hpp"

using namespace std;



class IcsiboostParameterType {

    private:

    public:
        IcsiboostParameterType() {};
        virtual int getNeededNeurons() throw (SfannException) {return 0;};
        virtual fann_type * convertToNeuralRepresentation(const string & icsi_data) throw (SfannException) {return NULL;};
        virtual string toString() {return string("nothing");};
        virtual ~IcsiboostParameterType() {};
};


class IcsiboostParameterLabels : public IcsiboostParameterType {

    private:
        map <string, int> label2id;
        map <int, string> id2label;

    public:

        IcsiboostParameterLabels(const string & icsi_param_description);
        void setLabels(const string & icsi_param_description);
        int getNeededNeurons() throw (SfannException);
        fann_type * convertToNeuralRepresentation(const string & icsi_data) throw (SfannException);
        string toString();
};

// class IcsiboostNames;

class IcsiboostNames {

    private:
//         map <string, int> labels;
        string file_path;

        IcsiboostParameterLabels * labels;
        map <int, IcsiboostParameterType*> parameters;
        map <int, string> parameterNames;
        map <string, int> parameterNums;
        int neededNeurons;

        void loadFile(const string & file) throw (SfannException);
        static bool isCommentLine(const string & line);
        static void readNext(ifstream & f, string & line, int & num_line, bool skipComments);

    public:

        IcsiboostNames();
        IcsiboostNames(const string & file) throw (SfannException);

        string getParameterName(int num);
        int getParameterNum(string name);
        IcsiboostParameterType * getParameter(string name);
        IcsiboostParameterType * getParameter(int num);
        IcsiboostParameterType * getLabels();
        int getNbParameters();

        int getNeededNeurons();

        string toString();
};


class IcsiboostDataParser {
    private:
    public:
        static fann_type * convertIcsiExempleToFannInput(const string & exemple, IcsiboostNames & names);
        static fann_type * convertIcsiExempleToFannOutput(const string & exemple, IcsiboostNames & names);
        static struct fann_train_data * loadDataToFann(const string & file, IcsiboostNames & names) throw (SfannException);
};




class IcsiboostParameterContinuous : public IcsiboostParameterType {

    private:

    public:
        IcsiboostParameterContinuous();
        int getNeededNeurons() throw (SfannException);
        fann_type * convertToNeuralRepresentation(const string & icsi_data) throw (SfannException);
        string toString();
};



class IcsiboostParameterFactory {
	private:

	public:
		static IcsiboostParameterType * createIcsiboostParameter(const string & icsi_def_line) throw (SfannException);
		static string stripString(const string & str, const char * sep);
};


class IcsiboostUtils {
    public:
        static void tokenize(const string& str, vector<string>& tokens, const string& delimiters, bool strip_spaces);
        static void stripSpacePositions(const string& str, size_t & deb, size_t & fin);
};



#endif








