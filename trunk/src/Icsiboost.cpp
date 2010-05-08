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

#include "Icsiboost.hpp"


using namespace std;

//  Names

IcsiboostNames::IcsiboostNames() {
    this->labels = NULL;
    this->neededNeurons = 0;
}

IcsiboostNames::IcsiboostNames(const string & file) throw (SfannException) {
    this->labels = NULL;
    this->neededNeurons = 0;
    this->loadFile(file);
}

int IcsiboostNames::getNeededNeurons() {
    return this->neededNeurons;
}

void IcsiboostNames::loadFile(const string & file) throw (SfannException) {

    ifstream f(file.c_str());
    if (f.is_open()) {
        if (this->labels != NULL) {
            delete this->labels;
            this->labels = NULL;
        }
        this->parameters.clear();
        this->parameterNames.clear();
        this->parameterNums.clear();

        this->file_path = file;
        int num_line = 0;
        string line;
        getline(f, line);
        num_line++;

        this->labels = new IcsiboostParameterLabels(line);

        int num_parameter = 0;
        while(! f.eof()) {
            getline(f, line);
            num_line++;
            // skip comments and blank lines
            size_t diese = line.find_first_of("#");
            if (line.size() > 0 && (diese == string::npos || diese > line.find_first_not_of(" #\t"))) {
                vector<string> tokens;
                IcsiboostUtils::tokenize(line, tokens, ":", true);
                if (tokens.size() < 2) {
                    ostringstream oss;
                    oss << "Error in " << file << ": format of line " << num_line << " incorrect !";
                    throw *new SfannException(oss.str());
                }
                if (this->parameterNums.count(tokens.front()) > 0) {
                    ostringstream oss;
                    oss << "Error in " << file << ": duplicated parameter names at line " << num_line << " !";
                    throw *new SfannException(oss.str());
                }
                this->parameters[num_parameter] = IcsiboostParameterFactory::createIcsiboostParameter(tokens.back());
                this->parameterNums[tokens.front()] = num_parameter;
                this->parameterNames[num_parameter] = tokens.front();
                this->neededNeurons += this->parameters[num_parameter]->getNeededNeurons();
                num_parameter++;
            }
        }
    } else {
        ostringstream oss;
        oss << "Impossible read of " << file << " !";
        throw *new SfannException(oss.str());
    }
}

string IcsiboostNames::getParameterName(int num) {
    return this->parameterNames[num];
}

int IcsiboostNames::getParameterNum(string name) {
    return this->parameterNums[name];
}

IcsiboostParameterType * IcsiboostNames::getParameter(string name) {
    if (this->parameterNums.count(name) > 0 && this->parameters.count(this->parameterNums[name]) > 0)
        return this->parameters[this->parameterNums[name]];
    else
        return NULL;
}

IcsiboostParameterType * IcsiboostNames::getLabels() {
    return this->labels;
}

IcsiboostParameterType * IcsiboostNames::getParameter(int num) {
    if (this->parameters.count(num) > 0)
        return this->parameters[num];
    else
        return NULL;
}

int IcsiboostNames::getNbParameters() {
    return this->parameters.size();
}

string IcsiboostNames::toString() {
    if (this->labels != NULL) {
        string res;
        res += "Detected " + this->labels->toString() + "\n";
        res += "Parameters :\n";
        for (map <int, IcsiboostParameterType*>::iterator it = this->parameters.begin(); it != this->parameters.end(); ++it) {
            res += "    - " + this->parameterNames[it->first] + " : " + it->second->toString() + "\n";
        }
        return res;
    } else {
        return string("No file loaded");
    }
}

// IcsiboostDataParser


fann_type * IcsiboostDataParser::convertIcsiExempleToFannInput(const string & exemple_line, IcsiboostNames & names) {
    fann_type * res = new fann_type[names.getNeededNeurons()];
    int ires = 0;

    vector<string> param_vals;
    IcsiboostUtils::tokenize(exemple_line, param_vals, ",", true);
    if (param_vals.size() != names.getNbParameters()+1) {
        throw *new SfannException("Bad number of parameter values in data line ("+exemple_line+")");
    }
    
    for (int i=0; i < param_vals.size()-1; ++i) {
        fann_type * tmp = names.getParameter(i)->convertToNeuralRepresentation(param_vals[i]);
        for (int j=0; j<names.getParameter(i)->getNeededNeurons(); j++) {
            res[ires++] = tmp[j];
        }
        delete tmp;
    }
    
    return res;
}

fann_type * IcsiboostDataParser::convertIcsiExempleToFannOutput(const string & exemple_line, IcsiboostNames & names) {
    size_t deb = exemple_line.find_last_of(",");
    size_t fin = exemple_line.find_last_of(".");
    IcsiboostUtils::stripSpacePositions(exemple_line, deb, fin);
    return names.getLabels()->convertToNeuralRepresentation(exemple_line.substr(deb, fin-deb));
}

struct fann_train_data * IcsiboostDataParser::loadDataToFann(const string & file, IcsiboostNames & names) throw (SfannException) {
    vector<string> file_content;
    
    ifstream f(file.c_str());
    if (f.is_open()) {
        string line;
        int num_line = 0;
        while(! f.eof()) {
            getline(f, line);
            num_line++;
            // skip comments and blank lines
            size_t diese = line.find_first_of("#");
            if (line.size() > 0 && (diese == string::npos || diese > line.find_first_not_of(" #\t"))) {
                file_content.push_back(line);
            }
        }
    } else {
        ostringstream oss;
        oss << "Impossible read of " << file << " !";
        throw *new SfannException(oss.str());
    }

    int nb_exemples = file_content.size();
    int nb_input = names.getNeededNeurons();
    int nb_output = names.getLabels()->getNeededNeurons();

    struct fann_train_data * res = new struct fann_train_data;
    
    res->errno_f = FANN_E_NO_ERROR;
    res->error_log = NULL;
    res->errstr = NULL;
    res->num_input = nb_input;
    res->num_output = nb_output;
    res->num_data = nb_exemples;

    if (nb_exemples <= 0) {
        res->input = NULL;
        res->output = NULL;
    } else {
        res->input = new fann_type * [nb_exemples];
        for (int i=0; i<nb_exemples; i++) res->input[i] = IcsiboostDataParser::convertIcsiExempleToFannInput(file_content[i], names);

        res->output = new fann_type * [nb_exemples];
        for (int i=0; i<nb_exemples; i++) res->output[i] = IcsiboostDataParser::convertIcsiExempleToFannOutput(file_content[i], names);
    }

    return res;
}


//  ParameterContinuous

IcsiboostParameterContinuous::IcsiboostParameterContinuous() {

}

int IcsiboostParameterContinuous::getNeededNeurons() throw (SfannException) {
    return 1;
}

string IcsiboostParameterContinuous::toString() {
	return string("continuous");
}

fann_type * IcsiboostParameterContinuous::convertToNeuralRepresentation(const string & icsi_data) throw (SfannException) {
    fann_type * res = new fann_type[1];
    sscanf(icsi_data.c_str(), "%f", res);

/*    istringstream iss (icsi_data);
    double t;
    iss >> t;
    res[0] = t;
    if (! iss.good()) {
        ostringstream oss;
        oss << "Impossible conversion of " << icsi_data << " to fann_type !";
        throw *new SfannException(oss.str());
    }*/

    return res;
}


// ParameterLabels

IcsiboostParameterLabels::IcsiboostParameterLabels(const string & icsi_param_description) {
    this->setLabels(icsi_param_description);
}

void IcsiboostParameterLabels::setLabels(const string & icsi_param_description) {
    this->label2id.clear();
    this->id2label.clear();
    
    vector<string> tokens;
    IcsiboostUtils::tokenize(icsi_param_description, tokens, ".,", true);

    for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); ++it) {
        if (this->label2id.count((*it)) <= 0) {
            int id = this->label2id.size();
            this->label2id[(*it)] = id;
            this->id2label[id] = (*it);
        }
    }
}

string IcsiboostParameterLabels::toString() {
	ostringstream oss;
	oss << "Labels: ";
	for (map<int, string>::iterator it = this->id2label.begin();
		it != this->id2label.end();
		++it) {
		oss << it->first << "=" << it->second << " ";
	}
	return oss.str();
}


int IcsiboostParameterLabels::getNeededNeurons() throw (SfannException) {
    return this->label2id.size();
}

fann_type * IcsiboostParameterLabels::convertToNeuralRepresentation(const string & icsi_data) throw (SfannException) {
    fann_type * res = new fann_type[this->label2id.size()];
    for(int i=0; i<this->label2id.size(); i++)
        res[i] = -1;

    vector<string> tokens;
    IcsiboostUtils::tokenize(icsi_data, tokens, ",.", true);

    for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); ++it) {
        if (this->label2id.count(*it) > 0) {
            res[this->label2id[(*it)]] = 1;
        }
    }
    
    return res;
}


// IcsiboostUtils

void IcsiboostUtils::tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ", bool strip_spaces = false){
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        size_t p = pos, l = lastPos;
        // strip
        if (strip_spaces) IcsiboostUtils::stripSpacePositions(str, l, p);
        tokens.push_back(str.substr(l, p - l));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

void IcsiboostUtils::stripSpacePositions(const string& str, size_t & deb, size_t & fin) {
    while(fin > 0 && str[fin-1] == ' ') fin--;
    while(deb < str.size() && str[deb] == ' ') deb++;
}


// IcsiboostParameterFactory 

IcsiboostParameterType * IcsiboostParameterFactory::createIcsiboostParameter(const string & icsi_def_line) throw (SfannException) {
	string stripped = IcsiboostParameterFactory::stripString(icsi_def_line, "\t ");
	if (stripped.compare("continuous.") == 0) {
//         cout << "Create continuous !" << endl;
		return new IcsiboostParameterContinuous();  
	} else if (stripped.compare("text.") == 0) {
//         cout << "Create text !" << endl;
		return new IcsiboostParameterType();
	} else {
//         cout << "Create labels !" << endl;
		return new IcsiboostParameterLabels(stripped);
	}
}

string IcsiboostParameterFactory::stripString(const string & str, const char * sep) {
	size_t const first = str.find_first_not_of(sep);
	return (first == string::npos ? string() : str.substr(first, str.find_last_not_of(sep)-first+1));
}



