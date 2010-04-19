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

#include "SfannException.hpp"

SfannException::SfannException(string m) throw() {
    this->message = m;
}

SfannException::SfannException(const SfannException &source) throw() {
    this->message = source.message;
}

SfannException::~SfannException() throw() {
        
}

const char * SfannException::what() const throw() {
    return this->message.c_str();
}
