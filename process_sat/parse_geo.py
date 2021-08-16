'''
Framework for parsing geolocated satellite files

Instantiates appropriate class with filename and
optional strings specifying the subtype of file and 
an override extension.

All classes store the following parameters:
    name - the full pathname of the file
    ext - the extension of the file
    sub - the subtype of the file

Variables in the file are accessible through the get method
which optionally retrieves individual indices.  The optional 
parameter indices should specify which indices to retrieve
in a tuple.  Multidimensional variables are retrieved as 
slices along the "common" dimensions (not those unique to the
 variable).  Not passing indices will result in the retrieval
 of all the data from that variable, though the dimensionality
 of that data is not guaranteed.  Returns nan's for
 missing data values ONLY FOR TYPES THAT START AS FLOATS.  Int types 
keep their original fill value, but it it scaled and offset along
with everything else (so if you want to test for it, apply the
scale and offset to the fill value before testing)

Parser is responsible for properly applying scale-offset
to retrieved data.  

Parsers should throw IOError if passed an invalid file

There are several special methods that must return specific
data regardless of filetype.  These methods are used
by mapping functions, so they only have to be implmented
if the desired mapping function requires them.
    get_geo_corners() - returns a record array with three fields.
                        The first field, lat, contains 4 floats, 
                        the latitudes of the 4 corners.  The field
                        lon contains 4 floats, the longitudes of 
                        the 4 corners.  The field ind is as 
                        large as it needs to be to contain the 
                        indices.  each ind should be castable
                        to a tuple that can be fed into get.
    get_geo_centers() - returns a record arraw with 3 fields.  The
                        first field, lat, contains the latitude of
                        the pixel center.  The lon field contains
                        the longitude of the pixel center.  The
                        field ind is as alarge as it needs to be
                        to contain the indices.  If cast to a 
                        tuple and fed into the get() function,
                        it should retrieve the same pixel
                        
The following functions may be implemented or not in any class.
They duplicate the functionality of the get function but in
some cases may allow for much more efficient operation.  If 
they are not implemented, they are expected to throw a
NotImplementedError when called.
    __enter__()                - enter method for context manager
    __exit__()                 - exit method for context manager
    get_cm(key, ind=(:)) - an alternate form of the get statement meant to be
                 used inside of a context manager (with statement).
                 Requiring use of a context manager enables 
                 any files to be left open, improving efficiency.
                 While it is not required, it is recommended that this
                 method throw some kind of error when called outside a
                 context manager.  Must operate exactly the same as
                 the get function in terms of inputs and output.
                 
This framework can be extended by adding classes for particular (sub)class
'''

import os
import sys
import string
import pdb

import tables
import numpy
import pyhdf.HDF
import pyhdf.V
import pyhdf.VS
import pyhdf.SD

from builtins import str
from builtins import range
from builtins import object 
from netCDF4 import Dataset # Eliot addition
from . import filetypes

def SupportedFileTypes():
    '''Return a list of supported file types'''
    return [el[:-9] for el in dir(filetypes) if el.endswith("_filetype")]


def getOrbitNumber(fPath):    
    '''Takes in the path to a nasa omi hdf file and returns the orbit number'''
    fid = tables.open_file(fPath)
    try:
        node = fid.get_node('/', 'HDFEOS INFORMATION/CoreMetadata')
    except tables.exceptions.NoSuchNodeError:
        node = fid.get_node('/', 'HDFEOS INFORMATION/CoreMetadata.0')
    bigString = str(list(node)[0])
    strings = bigString.split('\\n')
    for i in range(len(strings)):
        if 'ORBITNUMBER' in strings[i]:
            break
    line = strings[i+3]
    numArray = [int(el) for el in line.split() if el.isdigit()]
    fid.close()
    return numArray[0]

def getLongName(fPath):
    '''Retrieve the long name of an HDFEOS file'''
    fid = tables.open_file(fPath)
    try:
        node = fid.get_node('/', 'HDFEOS INFORMATION/ArchiveMetadata')
    except tables.exceptions.NoSuchNodeError:
        try:
            node = fid.get_node('/', 'HDFEOS INFORMATION/ArchivedMetadata.0')
        except:
            node = fid.get_node('/', 'HDFEOS INFORMATION/ArchivedMetadata')
    bigString = str(list(node)[0])
    strings = bigString.split('\\n')
    for i in range(len(strings)):
        # if 'LONGNAME' in strings[i]:
            # break
        if 'VALUE' in strings[i]:
            break
    chunks = strings[i].split('=')[1].split('"')[1]
    #line = strings[i+2]
    #chunks = line.split('"')
    fid.close()
    return chunks

def get_parser(file, filetype, parserParms):
    """Retrieve appropriate instantiated parser for a file"""
    # filename = os.path.split(file)[1]
    subclass = '{0}_File'.format(filetype)
    module = sys.modules[GeoFile.__module__]
    parserClass = getattr(module, subclass) 
                  # or GeoFile
    extension = ''
    subtype = ''
    for i in filetype:
        if subtype == '' and i in string.ascii_uppercase:
            extension += i
        else:
            subtype += i
    return parserClass(file, subtype, extension, **parserParms)

class GeoFile(object):
    """Provide interface to geofile."""
    def __init__(self, filename, subtype='', extension=None):
        self.name = filename
        self.ext = extension or os.path.splitext(filename)[1][1:]
        self.sub = subtype
    def get(self, key, indices=None):
        raise NotImplementedError
    def get_geo_corners(self):
        raise NotImplementedError
    def get_geo_centers(self):
        raise NotImplementedError
    def __enter__(self):
        raise NotImplementedError
    def __exit__(self):
        raise NotImplementedError
    def get_cm(self, key, indices=None):
        raise NotImplementedError

class HDF4File(GeoFile):
    """Provide generic interface for HDF 4 files"""
    def __init__(self, filename, subtype='', extension=None):
        GeoFile.__init__(self, filename, subtype=subtype, extension=extension)
        if pyhdf.HDF.ishdf(self.name):
            pass
        else:
            raise IOError('Attempt to read non HDF4 file as HDF4')

    def walkHDF4(self, fid, pathList, vInt, vsInt, sdInt):
        """
        Retrives a variable or variable group from an HDF4 file

        Requires the file handle fid and a list that contains each
        element of the path to the leaf that we want.  Returns the leaf,
        which may be of any type.

        Assumes that the leaf we want is a Vgroup or Vdata datatype.

        Arguments vInt and vsInt are the VG and VS interfaces
        as defined by pyhdf.  If passed, they  WILL NOT
        be closed.  

        To repeat, if the interfaces are passed in they will NOT be
        safely closed.
        """
        leafName = pathList[-1] # name of the leaf we want
        # get it the easy way if it's a scientific dataset
        sciData = sdInt.datasets()
        if leafName in sciData:
            return sdInt.select(leafName)
        # it must not be a scientific dataset, so walk the file to find it
        pList = list(pathList) # shallow Copy
        parent = vInt.attach(vInt.getid(-1))
        pName = pList.pop(0)
        if parent._name != pName:
            raise AttributeError("Bad data path (did not start at root).")
        while parent._name != leafName:
            cName = pList.pop(0)
            children = parent.tagrefs()
            for (childType, childRef) in children:
                
                if childType == pyhdf.HDF.HC.DFTAG_VG:
                    child = vInt.attach(childRef)
                elif childType == pyhdf.HDF.HC.DFTAG_VH:
                    child = vsInt.attach(childRef)
                elif childType == pyhdf.HDF.HC.DFTAG_NDG:
                    # we know this can't be it so keep looking
                    continue
                else:
                    raise IOError('Unknown data format.  Check data structure.')

                if child._name == cName:
                    parent.detach()
                    parent = child
                    break
                else:
                    child.detach()
                    
            if parent is not child:
                raise AttributeError('Bad data path.  Check parser/data structure.')

        return parent

    def get(self, key, indices=None, missingValue=None):
        """
        Provide get functionality for HDF 4 files.  

        Assumes absolutely no attributes present.

        If missingValue is provided, it will be used to mask floating
        point data properly with NaN's.  If it is not provided, the data
        will be returned as is, -9999.0's and all.

        Requires that parser be set up with _nameExpMap and _indexMap 
        variables.  These must be defined as:
        
        _nameExpMap -   Dictionary.  Keys are field names available to user.
                        Values are the full path to that field in the HDF file.
                       
        _indexMap -     Dictionary.  Keys are field names available to user.  
                        Values are functions that when passed (var, ind) where
                        ind is a n-element tuple will return the proper slice.
                        n is the number of fundamental dimensions of the 
                        file type.
        """
        fid = pyhdf.HDF.HDF(self.name)
        try:
            vInt = fid.vgstart()
            vsInt = fid.vstart()
            sdInt = pyhdf.SD.SD(self.name)
            path = self._nameExpMap[key]
            pathList = [el for el in path.split('/') if el] # path list with no empty strings
            vNode = self.walkHDF4(fid, pathList, vInt, vsInt, sdInt)
            vData = numpy.array(vNode[:])
        except AttributeError as err:
            raise IOError("No field %s.  May be attempt to read non-MOPPIT file as such." % self._nameExpMap[key])
        except KeyError:
            raise IOError("Attempt to use fieldname not associated with this filetype.")
        finally:
            # clean up from the bottom up
            try:
                vNode.detach()
            except(NameError):
                pass
            except(AttributeError):
                # must have been a scientific dataset
                vNode.endaccess()

            try:
                sdInt.end()
            except(NameError):
                pass

            try:
                vsInt.end()
            except(NameError):
                pass

            try:
                vInt.end()
            except(NameError):
                pass

            fid.close
        
        # convert missing values if appropriate
        if missingValue and vData.dtype in ['float32', 'float64', 'int16']:
            vData = numpy.where(vData == missingValue, numpy.NaN, vData)

        # use indices if we have them
        if indices is not None:
            # we want specific indices, use _indexMap
            indFunc = self._indexMap.get(key, self._indexMap['default'])
            return indFunc(vData, indices)
        else:
            # just fetch everything
            return vData

    def __enter__(self):
        '''Open up file and leave open.'''
        self._fid = pyhdf.HDF.HDF(self.name)
        self._open_vars = dict()
        self._vsInt = self._fid.vstart()
        self._vInt = self._fid.vgstart()
        self._sdInt = pyhdf.SD.SD(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Close file and delete references to file object and nodes.'''
        self._sdInt.end()
        self._vsInt.end()
        self._vInt.end()
        self._fid.close()
        del self._open_vars
        del self._fid
        del self._vsInt
        del self._vInt
        return False

    def get_cm(self, key, indices=None, missingValue=None):
        """
        Provide get_cm function for HDF files

        get_cm works the same as get, but relies on a context manager to speed
        up access by allowing it safely leave open variables in memory.

        Assumes absolutely no attributes present.

        If missingValue is provided, it will be used to mask floating
        point data properly with NaN's.  If it is not provided, the data
        will be returned as is, -9999.0's and all.

        Requires that parser be set up with _nameExpMap and _indexMap 
        variables.  These must be defined as:
        
        _nameExpMap -   Dictionary.  Keys are field names available to user.
                        Values are the full path to that field in the HDF file.
                       
        _indexMap -     Dictionary.  Keys are field names available to user.  
                        Values are functions that when passed (var, ind) where
                        ind is a n-element tuple will return the proper slice.
                        n is the number of fundamental dimensions of the 
                        file type.
        """
        # open the variable if it isn't open already.
        if key not in list(self._open_vars.keys()):
            try:
                path = self._nameExpMap[key]
                pathList = [el for el in path.split('/') if el] 
                vNode = self.walkHDF4(self._fid, pathList, self._vInt, self._vsInt, self._sdInt)
                self._open_vars[key] = numpy.array(vNode[:])
                try:
                    vNode.detach()
                except AttributeError:
                    # must have been a scientific dataset
                    vNode.endaccess()
            except AttributeError:
                raise IOError("No field %s.  May be attempt to read non-MOPPIT file as such." % self._nameExpMap[key])
            except KeyError:
                raise IOError("Attempt to use fieldname %s, which is not associated with this filetype." % key)
                
            # convert missing values if appropriate
            if missingValue and self._open_vars[key].dtype in ['float32', 'float64']:
                self._open_vars[key] = numpy.where(self._open_vars[key] == missingValue,
                                                   numpy.NaN, self._open_vars[key])

        # retrieve value of interest from the (newly?) open variable
        if indices is not None:
            # we want specific indices, use _indexMap
            indFunc = self._indexMap.get(key, self._indexMap['default'])
            return indFunc(self._open_vars[key], indices)
        else:
            # just fetch everything
            return self._open_vars[key]
            
class HDFFile(GeoFile):
    """Provide generic interface for HDF 5 files"""
    def __init__(self, filename, subtype='', extension=None):
        GeoFile.__init__(self, filename, subtype=subtype, extension=extension)
        if tables.is_hdf5_file(self.name):  # sanity check
            pass
        else:
            raise IOError('Attempt to read non-HDF 5 file as HDF 5.')
        
    def get(self, key, indices=None):
        """
        Provide get function for HDF Files.
        
        Requires that parser be set up with _nameExpMap and _indexMap 
        variables.  These must be defined as:
        
        _nameExpMap -   Dictionary.  Keys are field names available to user.
                        Values are the full path to that field in the HDF file.
                       
        _indexMap -     Dictionary.  Keys are field names available to user.  
                        Values are functions that when passed (var, ind) where
                        ind is a n-element tuple will return the proper slice.
                        n is the number of fundamental dimensions of the 
                        file type.
        """
        fid = tables.open_file(self.name)
        try:
            var = fid.get_node('/', self._nameExpMap[key])
            varAtts = var._v_attrs
            missing = getattr(varAtts, '_FillValue', numpy.nan)
            # because attributes are single element arrays
            # and not zero-element arrays, they change the 
            # rank of return values when applied. We take 
            # the first element to get zero-rank arrays
            scale = getattr(varAtts,'ScaleFactor', [1.0])[0]
            offset = getattr(varAtts, 'Offset', [0.0])[0]
            if var[:].dtype in ['float32', 'float64']:
                # only cast if we have a type that features nans
                var = numpy.where(var == missing, numpy.NaN, var)
                
            if indices is not None:
                # we want specific indices, use _indexMap
                indFunc = self._indexMap.get(key, self._indexMap['default'])  # fetch default if not in index map
            else:
                # don't bother with _indexMap, just fetch everything
                indFunc = lambda var, ind: var[:]
            
            if (scale != 1) or (offset != 0):  # avoids casting if we don't need to
                return indFunc(var[:], indices)*scale + offset
            else:
                return indFunc(var[:], indices)
        except (tables.exceptions.NoSuchNodeError, AttributeError):
            raise IOError("No field %s.  May be attempt to read non-KNMI Aura OMI file as such." % self._nameExpMap[key])
        except KeyError:
            raise KeyError("Attempt to use fieldname not associated with this filetype.")
        finally:
            fid.close()
                
    def get_cm(self, key, indices=None):
        """
        Provide get_cm function for HDF files
        
        get_cm works the same as get, but relies on a context manager to speed
        up access to the underlying files.  Just as get, it requires that 
        the parser have _nameExpMap and _indexMap variables.  These must be
        defined as above.
        """
        # open the var if it isn't open already
        if key not in list(self._open_vars.keys()):
            try:
                var = self._fid.get_node('/', self._nameExpMap[key])
                varAtts = var._v_attrs
                missing = getattr(varAtts, '_FillValue', numpy.nan)
                # because attributes are single element arrays
                # and not zero-element arrays, they change the 
                # rank of return values when applied. We take 
                # the first element to get zero-rank arrays   
                scale = getattr(varAtts, 'ScaleFactor', [1.0])[0]
                offset = getattr(varAtts, 'Offset', [0.0])[0]
                if var[:].dtype in ['float32', 'float64']:
                    # only do nan sub if we don't have to cast
                    var = numpy.where(var == missing, numpy.NaN, var)
                self._open_vars[key] = var
                self._scales[key] = scale
                self._offsets[key] = offset
            except(KeyError):
                raise KeyError("No variable " + key + " in file " + self.name)
            except(tables.exceptions.NoSuchNodeError, AttributeError):
                raise IOError("No field %s.  May be attempt to read non-KNMI Aura OMI file as such." % self._nameExpMap[key])
        
        # return the values from the open var
        
        if indices is not None:
            # we have indices, use _indexMap
            indFunc = self._indexMap.get(key, self._indexMap['default']) # fetch default if not index map
        else:
            # we want everything, don't bother with _indexMap
            indFunc = lambda var, ind: var[:]
        
        if (self._scales[key] != 1) or (self._offsets[key] != 0): 
            return (indFunc(self._open_vars[key], indices)
                    *self._scales[key]+self._offsets[key])
        else:
            return indFunc(self._open_vars[key], indices)        
            
    def __enter__(self):
        '''Open up file and leave open.'''
        self._fid = tables.open_file(self.name, mode='r')
        self._open_vars = dict()
        self._scales = dict()
        self._offsets = dict()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        '''Close file and delete references to file object and nodes'''
        self._fid.close()
        del self._open_vars
        del self._scales
        del self._offsets
        del self._fid
        return False

# Author: Eliot Kim
# Date: 8-11-2020
class NETCDFFile(GeoFile):
    """Provide generic interface for netCDF4 files"""
    def __init__(self, filename, subtype='', extension=None):
        GeoFile.__init__(self, filename, subtype=subtype, extension=extension)

    def get(self, key, indices=None):
        """
        Provide get function for netCDF Files.
        
        Requires that parser be set up with _nameExpMap and _indexMap 
        variables.  These must be defined as:
        
        _nameExpMap -   Dictionary.  Keys are field names available to user.
                        Values are the full path to that field in the netCDF file.
                       
        _indexMap -     Dictionary.  Keys are field names available to user.  
                        Values are functions that when passed (var, ind) where
                        ind is a n-element tuple will return the proper slice.
                        n is the number of fundamental dimensions of the 
                        file type.
        """
        fid = Dataset(self.name)
        try:
            var = fid[self._nameExpMap[key]]
            missing = var._FillValue

            if var[:].dtype in ["float32", "float64"]:
                var = numpy.where(var == missing, numpy.NaN, var)

            if indices is not None:
                indFunc = self._indexMap.get(key, self._indexMap["default"])
            else:
                indFunc = lambda var, ind: var[:]

            return indFunc(var[:], indices)
        except KeyError:
            raise KeyError("Attempt to use fieldname not associated with this filetype.")
        finally:
            fid.close()

    def get_cm(self, key, indices=None):
        """
        Provide get_cm function for netCDF files
        
        get_cm works the same as get, but relies on a context manager to speed
        up access to the underlying files.  Just as get, it requires that 
        the parser have _nameExpMap and _indexMap variables.  These must be
        defined as above.
        """
        if key not in list(self._open_vars.keys()):
            try:
                # print(key)
                var = self._fid[self._nameExpMap[key]]
                # print(var, var.shape, indices)
                try:
                    missing = var._FillValue
                    if var[:].dtype in ["float32", "float64"]:
                        var = numpy.where(var == missing, numpy.NaN, var)
                except AttributeError:
                    print("no fill value attribute")
                    pass
                self._open_vars[key] = var

            except KeyError:
                raise KeyError("Attempt to use fieldname not associated with this filetype.")

        if indices is not None:
            indFunc = self._indexMap.get(key, self._indexMap["default"])
        else:
            indFunc = lambda var, ind: var[:]

        if key == "tm5_pressure_level_a" or key == "tm5_pressure_level_b" or 'tm5_constant_a' or 'tm5_constant_b':
            return indFunc(self._open_vars[key], indices)

        return indFunc(self._open_vars[key], indices)[0]

    def __enter__(self):
        self._fid = Dataset(self.name, "r")
        self._open_vars = dict()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._fid.close()
        del self._open_vars
        del self._fid
        return False

class HDFknmiomil2_File(HDFFile):
    """Provide interface to KNMI OMI L2 NRT product"""
    _nameExpMap = {"AirMassFactor"                          : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AirMassFactor",
                  "AirMassFactorGeometric"                  : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AirMassFactorGeometric",
                  "AirMassFactorTropospheric"               : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AirMassFactorTropospheric",
                  "AssimilatedStratosphericSlantColumn"     : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AssimilatedStratosphericSlantColumn",
                  "AssimilatedStratosphericVerticalColumn"  : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AssimilatedStratosphericVerticalColumn", 
                  "AveragingKernel"                         : "/HDFEOS/SWATHS/DominoNO2/Data Fields/AveragingKernel",
                  "CloudFraction"                           : "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudFraction",
                  "CloudFractionStd"                        : "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudFractionStd",
                  "CloudPressure"                           : "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudPressure",
                  "CloudPressureStd"                        : "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudPressureStd",
                  "CloudRadianceFraction"                   : "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudRadianceFraction",
                  "GhostColumn"                             : "/HDFEOS/SWATHS/DominoNO2/Data Fields/GhostColumn",
                  "InstrumentConfigurationId"               : "/HDFEOS/SWATHS/DominoNO2/Data Fields/InstrumentConfigurationId",
                  "MeasurementQualityFlags"                 : "/HDFEOS/SWATHS/DominoNO2/Data Fields/MeasurementQualityFlags",
                  "SlantColumnAmountNO2"                    : "/HDFEOS/SWATHS/DominoNO2/Data Fields/SlantColumnAmountNO2",
                  "SlantColumnAmountNO2Std"                 : "/HDFEOS/SWATHS/DominoNO2/Data Fields/SlantColumnAmountNO2Std",
                  "SurfaceAlbedo"                           : "/HDFEOS/SWATHS/DominoNO2/Data Fields/SurfaceAlbedo",
                  "TM4PressurelevelA"                       : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TM4PressurelevelA",
                  "TM4PressurelevelB"                       : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TM4PressurelevelB",
                  "TM4SurfacePressure"                      : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TM4SurfacePressure",
                  "TM4TerrainHeight"                        : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TM4TerrainHeight",
                  "TM4TropoPauseLevel"                      : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TM4TropoPauseLevel",
                  "TerrainHeight"                           : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TerrainHeight", # random fact: _FillValue attribute is inaccurate for this field
                  "TotalVerticalColumn"                     : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TotalVerticalColumn",
                  "TotalVerticalColumnError"                : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TotalVerticalColumnError",
                  "TroposphericColumnFlag"                  : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TroposphericColumnFlag",
                  "TroposphericVerticalColumn"              : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TroposphericVerticalColumn",
                  "TroposphericVerticalColumnError"         : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TroposphericVerticalColumnError",
                  "TroposphericVerticalColumnModel"         : "/HDFEOS/SWATHS/DominoNO2/Data Fields/TroposphericVerticalColumnModel",
                  "VCDErrorUsingAvKernel"                   : "/HDFEOS/SWATHS/DominoNO2/Data Fields/VCDErrorUsingAvKernel",
                  "VCDTropErrorUsingAvKernel"               : "/HDFEOS/SWATHS/DominoNO2/Data Fields/VCDTropErrorUsingAvKernel",
                  "GroundPixelQualityFlags"                 : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/GroundPixelQualityFlags",
                  "Latitude"                                : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Latitude",
                  "LatitudeCornerpoints"                    : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/LatitudeCornerpoints",
                  "Longitude"                               : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Longitude",
                  "LongitudeCornerpoints"                   : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/LongitudeCornerpoints",
                  "SolarAzimuthAngle"                       : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/SolarAzimuthAngle",
                  "SolarZenithAngle"                        : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/SolarZenithAngle",
                  "Time"                                    : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Time",
                  "ViewingAzimuthAngle"                     : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/ViewingAzimuthAngle",
                  "ViewingZenithAngle"                      : "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/ViewingZenithAngle"    
               }
    _indexMap = {"default"                      : lambda var, ind: var[..., ind[0], ind[1]], 
                 "InstrumentConfigurationId"    : lambda var, ind: var[ind[0]],
                 "MeasurementQualityFlags"      : lambda var, ind: var[ind[0]],
                 "TM4PressurelevelA"            : lambda var, ind: var[:],
                 "TM4PressurelevelB"            : lambda var, ind: var[:],
                 "Time"                         : lambda var, ind: var[ind[0]]
                 }
    
    def get_geo_corners(self):
        lat = self.get('LatitudeCornerpoints')
        lon = self.get('LongitudeCornerpoints')
        lat = numpy.transpose(lat, (1,2,0))
        lon = numpy.transpose(lon, (1,2,0))
        ind = numpy.indices(lat.shape[0:2]).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
    
    def get_geo_centers(self):
        lat = self.get('Latitude')
        lon = self.get('Longitude')
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape,dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

        
class HDFnasaomil2_File(HDFFile):
    """
    Provide interface to NASA OMI L2 product, with pixel corners
    
    Pixel corners are retrieved from an extra file that must be accessible
    in cornerDir.  The files listed in cornerFileList, if any, will be checked
    first, followed by any files in cornerDir. If no valid pixel corner file
    is found that matches the orbit number of the input file, the parser will
    instantiage but get_geo_corners will fail with an IOError. 

    The corners retrieved are for the visible channel
    used by the NO2 algorithm- using this parser for other products may 
    require altering the parser to use a different channel if 
    appropriate.
    
    The keys to retrieve variables are the names of the variables within actual
    files.  Note that the NASA product documentation has the wrong name for 
    the "SlantColumnAmountH20Std" variable due to a case typo.

    Does not support the fields "UnpolFldCoefficients" or "SmallPixelRadiance"
    because the dimensionality of these fields changes between files and there
    is no way to deal with fields with variable dimension size in the current 
    framework.
    """
    
    OMIAURANO2_FILE_NAME = "OMI/Aura Nitrogen Dioxide (NO2) Total & "\
                           "Troposph.Column 1-Orbit L2 Swath 13x24km"
    OMIAURANO2_CORNER_FILE_NAME = "OMI/Aura Global Ground Pixel Corners "\
                                  "1-Orbit L2 Swath 13x24km"

    def __init__(self, filename, subtype='', extension=None, cornerDir=None,
                 cornerFileList=None):
        HDFFile.__init__(self, filename, subtype, extension)

        # make sure filename is actually an input file
        if getLongName(filename) != HDFnasaomil2_File.OMIAURANO2_FILE_NAME:
            raise IOError('Attempt to read non-NASA OMI L2 file as such.')
        
        # start by assuming we aren't going to find anything
        self.pixCorners = None

        # see if the corner directory even exists.  If it doesn't, we obviously can't 
        # find a corner file
        if os.path.isdir(cornerDir):

            # convert the corner files into full pathnames
            # unless we were given null string (signal to search directory)
            if cornerFileList != ['']:
                cornerFileList = [os.path.join(cornerDir, f) for f in cornerFileList]

            # get orbit number of file for matching
            forbitnumber = getOrbitNumber(filename)

            # try using the list
        
            for f in cornerFileList:
                if f != '' and getLongName(f) == HDFnasaomil2_File.OMIAURANO2_CORNER_FILE_NAME:
                    try:
                        if getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass

            # if necessary, search entire corner file directory
            if self.pixCorners == None:        
                allPossible = [os.path.join(cornerDir, f) for f in os.listdir(cornerDir)]
                for f in allPossible:
                    try:
                        if tables.is_hdf5_file(f) \
                                and getLongName(f) == HDFnasaomil2_File.OMIAURANO2_CORNER_FILE_NAME \
                                and getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass
                
        if self.pixCorners == None:
            print("No valid corner file found for {0}.".format(filename))
            
    __dataPath = '/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields/'
    __geoPath = '/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields/'
    
    _nameExpMap = {'AMFInitial' : __dataPath+'AMFInitial',
                    'AMFInitialClear' : __dataPath+'AMFInitialClear',
                    'AMFInitialClearStd' : __dataPath+'AMFInitialClearStd',
                    'AMFInitialCloudy' : __dataPath+'AMFInitialCloudy',
                    'AMFInitialCloudyStd' : __dataPath+'AMFInitialCloudyStd',
                    'AMFInitialStd' : __dataPath+'AMFInitialStd',
                    'AMFPolluted' : __dataPath+'AMFPolluted',
                    'AMFPollutedClear' : __dataPath+'AMFPollutedClear',
                    'AMFPollutedClearStd' : __dataPath+'AMFPollutedClearStd',
                    'AMFPollutedCloudy' : __dataPath+'AMFPollutedCloudy',
                    'AMFPollutedCloudyStd' : __dataPath+'AMFPollutedCloudyStd',
                    'AMFPollutedStd' : __dataPath+'AMFPollutedStd',
                    'AMFPollutedToGround' : __dataPath+'AMFPollutedToGround',
                    'AMFPollutedToGroundStd' : __dataPath+'AMFPollutedToGroundStd',
                    'AMFQualityFlags' : __dataPath+'AMFQualityFlags',
                    'AMFUnpolluted' : __dataPath+'AMFUnpolluted',
                    'AMFUnpollutedClear' : __dataPath+'AMFUnpollutedClear',
                    'AMFUnpollutedClearStd' : __dataPath+'AMFUnpollutedClearStd',
                    'AMFUnpollutedCloudy' : __dataPath+'AMFUnpollutedCloudy',
                    'AMFUnpollutedCloudyStd' : __dataPath+'AMFUnpollutedCloudyStd',
                    'AMFUnpollutedStd' : __dataPath+'AMFUnpollutedStd',
                    'ChiSquaredOfFit' : __dataPath+'ChiSquaredOfFit',
                    'CloudFraction' : __dataPath+'CloudFraction',
                    'CloudFractionStd' : __dataPath+'CloudFractionStd',
                    'CloudPressure' : __dataPath+'CloudPressure',
                    'CloudPressureStd' : __dataPath+'CloudPressureStd',
                    'CloudRadianceFraction' : __dataPath+'CloudRadianceFraction',
                    'ColumnAmountNO2' : __dataPath+'ColumnAmountNO2',
                    'ColumnAmountNO2Std' : __dataPath+'ColumnAmountNO2Std',
                    'ColumnAmountNO2BelowCloud' : __dataPath+'ColumnAmountNO2BelowCloud',
                    'ColumnAmountNO2BelowCloudStd' : __dataPath+'ColumnAmountNO2BelowCloudStd',
                    'ColumnAmountNO2Initial' : __dataPath+'ColumnAmountNO2Initial',
                    'ColumnAmountNO2InitialStd' : __dataPath+'ColumnAmountNO2InitialStd',
                    'ColumnAmountNO2Polluted' : __dataPath+'ColumnAmountNO2Polluted',
                    'ColumnAmountNO2PollutedStd' : __dataPath+'ColumnAmountNO2PollutedStd',
                    'ColumnAmountNO2Trop' : __dataPath+'ColumnAmountNO2Trop',
                    'ColumnAmountNO2TropStd' : __dataPath+'ColumnAmountNO2TropStd',
                    'ColumnAmountNO2Unpolluted' : __dataPath+'ColumnAmountNO2Unpolluted',
                    'ColumnAmountNO2UnpollutedStd' : __dataPath+'ColumnAmountNO2UnpollutedStd',
                    'FitQualityFlags' : __dataPath+'FitQualityFlags',
                    'InstrumentConfigurationId' : __dataPath+'InstrumentConfigurationId',
                    'MeasurementQualityFlags' : __dataPath+'MeasurementQualityFlags',
                    'PolynomialCoefficients' : __dataPath+'PolynomialCoefficients',
                    'PolynomialCoefficientsStd' : __dataPath+'PolynomialCoefficientsStd',
                    'RingCoefficient' : __dataPath+'RingCoefficient',
                    'RingCoefficientStd' : __dataPath+'RingCoefficientStd',
                    'RootMeanSquareErrorOfFit' : __dataPath+'RootMeanSquareErrorOfFit',
                    'SlantColumnAmountH2O' : __dataPath+'SlantColumnAmountH2O',
                    'SlantColumnAmountH2OStd' : __dataPath+'SlantColumnAmountH2OStd',
                    'SlantColumnAmountNO2' : __dataPath+'SlantColumnAmountNO2',
                    'SlantColumnAmountNO2Std' : __dataPath+'SlantColumnAmountNO2Std',
                    'SlantColumnAmountO2O2' : __dataPath+'SlantColumnAmountO2O2',
                    'SlantColumnAmountO2O2Std' : __dataPath+'SlantColumnAmountO2O2Std',
                    'SlantColumnAmountO3' : __dataPath+'SlantColumnAmountO3',
                    'SlantColumnAmountO3Std' : __dataPath+'SlantColumnAmountO3Std',
                    'SmallPixelRadiance' : __dataPath+'SmallPixelRadiance',
                    'SmallPixelRadiancePointer' : __dataPath+'SmallPixelRadiancePointer',
                    'TerrainHeight' : __dataPath+'TerrainHeight',
                    'TerrainPressure' : __dataPath+'TerrainPressure',
                    'TerrainReflectivity' : __dataPath+'TerrainReflectivity',
                    'TropopausePressure' : __dataPath+'TropopausePressure',
                    'TropFractionUnpolluted' : __dataPath+'TropFractionUnpolluted',
                    'TropFractionUnpollutedStd' : __dataPath+'TropFractionUnpollutedStd',
                    'UnpolFldLatBandQualityFlags' : __dataPath+'UnpolFldLatBandQualityFlags',
                    'WavelengthRegistrationCheck' : __dataPath+'WavelengthRegistrationCheck',
                    'WavelengthRegistrationCheckStd' : __dataPath+'WavelengthRegistrationCheckStd',
                    'XTrackQualityFlags' : __dataPath+'XTrackQualityFlags',
                    'VcdQualityFlags' : __dataPath+'VcdQualityFlags',
                    'GroundPixelQualityFlags' : __geoPath+'GroundPixelQualityFlags',
                    'Latitude' : __geoPath+'Latitude',
                    'Longitude' : __geoPath+'Longitude',
                    'SolarAzimuthAngle' : __geoPath+'SolarAzimuthAngle',
                    'SolarZenithAngle' : __geoPath+'SolarZenithAngle',
                    'SpacecraftAltitude' : __geoPath+'SpacecraftAltitude',
                    'SpacecraftLatitude' : __geoPath+'SpacecraftLatitude',
                    'SpacecraftLongitude' : __geoPath+'SpacecraftLongitude',
                    'Time' : __geoPath+'Time',
                    'ViewingAzimuthAngle' : __geoPath+'ViewingAzimuthAngle',
                    'ViewingZenithAngle' : __geoPath+'ViewingZenithAngle'}
    
    _indexMap = {'default' : lambda var, ind: var[ind[0], ind[1], ...],
                  'SmallPixelRadiance' : lambda var, ind: var[:, ind[1]],
                  'SmallPixelRadiancePointer' : lambda var, ind: var[ind[0], :],
                  'InstrumentConfigurationId' : lambda var, ind: var[ind[0]],
                  'MeasurementQualityFlags' : lambda var, ind: var[ind[0]],
                  'WavelengthRegistrationCheck' : lambda var, ind: var[ind[0], :],
                  'WavelengthRegistrationCheckStd' : lambda var, ind: var[ind[0], :],
                  'UnpolFldLatBandQualityFlags' : lambda var, ind: var[:],
                  'Time' : lambda var, ind: var[ind[0]],
                  'SpacecraftLatitude' : lambda var, ind: var[ind[0]],
                  'SpacecraftLongitude' : lambda var, ind: var[ind[0]],
                  'SpacecraftAltitude' : lambda var, ind: var[ind[0]]}
    
    def get_geo_corners(self):
        '''
        Retrieves array of the corners of the pixels.  
        
        Throws IOError if no pixel corner file specified
        '''
        latNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLatitude'
        lonNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLongitude'
        try:
            pxFid = tables.open_file(self.pixCorners)
        except AttributeError:
            raise IOError('Unable to open pixel corners file.  Need pixel corners file to use corners')
        try:
            latNode = pxFid.get_node('/', latNodeName)
            lonNode = pxFid.get_node('/', lonNodeName)
            # Note: it is assumed that there are no missing values.
            lat = latNode[:].transpose((1,2,0))
            lon = lonNode[:].transpose((1,2,0))            
        finally:
            pxFid.close()
        ind = numpy.indices(lat.shape[0:2]).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
    
    def get_geo_centers(self):
        lat = self.get('Latitude')
        lon = self.get('Longitude')
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape, dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
        
class HDFmopittl2_File(HDF4File):
    """
    Provide interface to MOPITT level 2 V5 product

    Automatically sets the missing value for the data to
    -9999.0, as this is the missing value used (but not 
    documented) within the data.
    """
    __GeoPath = '/MOP02/Geolocation Fields/'
    __DataPath = '/MOP02/Data Fields/'

    _nameExpMap = {'Time' : __GeoPath + 'Time',
                   'Latitude' : __GeoPath + 'Latitude', 
                   'Longitude' : __GeoPath + 'Longitude',
                   'Seconds in Day' : '/MOP02/Data Fields/Seconds in Day',
                   'Pressure Grid' : '/MOP02/Data Fields/Pressure Grid',
                   'Solar Zenith Angle' : '/MOP02/Data Fields/Solar Zenith Angle',
                   'Satellite Zenith Angle' : '/MOP02/Data Fields/Satellite Zenith Angle',
                   'Surface Pressure' : '/MOP02/Data Fields/Surface Pressure',
                   'Retrieved Surface Temperature' : '/MOP02/Data Fields/Retrieved Surface Temperature',
                   'Retrieved Surface Emissivity' : '/MOP02/Data Fields/Retrieved Surface Emissivity',
                   'Retrieved CO Mixing Ratio Profile' : '/MOP02/Data Fields/Retrieved CO Mixing Ratio Profile',
                   'Retrieved CO Surface Mixing Ratio' : '/MOP02/Data Fields/Retrieved CO Surface Mixing Ratio',
                   'Retrieved CO Total Column' : '/MOP02/Data Fields/Retrieved CO Total Column',
                   'Retrieved CO Total Column Diagnostics' : '/MOP02/Data Fields/Retrieved CO Total Column Diagnostics',
                   'Retrieval Averaging Kernel Matrix' : '/MOP02/Data Fields/Retrieval Averaging Kernel Matrix',
                   'Retrieval Error Covariance Matrix' : '/MOP02/Data Fields/Retrieval Error Covariance Matrix',
                   'A Priori Surface Temperature' : '/MOP02/Data Fields/A Priori Surface Temperature',
                   'A Priori Surface Emissivity' : '/MOP02/Data Fields/A Priori Surface Emissivity',
                   'A Priori CO Mixing Ratio Profile' : '/MOP02/Data Fields/A Priori CO Mixing Ratio Profile',
                   'A Priori CO Surface Mixing Ratio' : '/MOP02/Data Fields/A Priori CO Surface Mixing Ratio',
                   'Level 1 Radiances and Errors' : '/MOP02/Data Fields/Level 1 Radiances and Errors',
                   'Degrees of Freedom for Signal' : '/MOP02/Data Fields/Degrees of Freedom for Signal',
                   'Surface Index' : '/MOP02/Data Fields/Surface Index',
                   'DEM Altitude' : '/MOP02/Data Fields/DEM Altitude',
                   'Cloud Description' : '/MOP02/Data Fields/Cloud Description',
                   'MODIS Cloud Diagnostics' : '/MOP02/Data Fields/MODIS Cloud Diagnostics',
                   'Water Vapor Climatology Content' : '/MOP02/Data Fields/Water Vapor Climatology Content',
                   'Retrieval Iterations' : '/MOP02/Data Fields/Retrieval Iterations',
                   'Information Content Index' : '/MOP02/Data Fields/Information Content Index',
                   'Signal Chi2' : '/MOP02/Data Fields/Signal Chi2',
                   'Swath Index' : '/MOP02/Data Fields/Swath Index'}
    _indexMap = {'default' : lambda var, ind: var[ind[0], ...],
                 'Pressure Grid' : lambda var, ind: var[:]}
        
    def get(self, key, indices=None):
        '''Overloaded version of get that applies the correct missing value.'''
        return HDF4File.get(self, key, indices, missingValue=-9999.0)

    def get_cm(self, key, indices=None):
        '''Overloaded version of get_cm that applied the correct missing value.'''
        return HDF4File.get_cm(self, key, indices, missingValue=-9999.0)

    def get_geo_centers(self):
        '''Retrieves array of the corners of the pixels'''
        lat = self.get('Latitude').squeeze()
        lon = self.get('Longitude').squeeze()
        ind = numpy.arange(lat.size).reshape(lat.size,1)
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, (1,))]
        struct = numpy.zeros(lat.size, dtype = protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct


class HDFmodisl2_File(HDF4File):
    """
    Provide interface to MODIS level 2 V5 product
    
    Uses the missing values provided in the _FVMap
    """
    __GeoPath = '/mod04/Geolocation Fields/'
    __DataPath = '/mod04/Data Fields/'

    _nameExpMap = {'Latitude' : __GeoPath+'Latitude', 
                   'Longitude' : __GeoPath+'Longitude',
                   'Scan_Start_Time' : __DataPath+'Scan_Start_Time',
                   'Solar_Zenith' : __DataPath+'Solar_Zenith',
                   'Solar_Azimuth' : __DataPath+'Solar_Azimuth',
                   'Sensor_Zenith' : __DataPath+'Sensor_Zenith',
                   'Sensor_Azimuth' : __DataPath+'Sensor_Azimuth',
                   'Cloud_Mask_QA' : __DataPath+'Cloud_Mask_QA',
                   'Scattering_Angle' : __DataPath+'Scattering_Angle',
                   'Optical_Depth_Land_And_Ocean' : __DataPath + \
                                 'Optical_Depth_Land_And_Ocean',
                   'Image_Optical_Depth_Land_And_Ocean' : __DataPath + \
                                 'Image_Optical_Depth_Land_And_Ocean',
                   'Optical_Depth_Ratio_Small_Land_And_Ocean' : __DataPath + \
                                 'Optical_Depth_Ratio_Small_Land_And_Ocean',
                   'Mean_Reflectance_Land_All' : __DataPath + \
                                 'Mean_Reflectance_Land_All',
                   'Standard_Deviation_Reflectance_Land_All' : __DataPath + \
                                 'Standard_Deviation_Reflectance_Land_All',
                   'Path_Radiance_Land' : __DataPath+'Path_Radiance_Land',
                   'Error_Path_Radiance_Land' : __DataPath + \
                                 'Error_Path_Radiance_Land',
                   'Critical_Reflectance_Land' : __DataPath + \
                                 'Critical_Reflectance_Land',
                   'Error_Critical_Reflectance_Land' : __DataPath + \
                                 'Error_Critical_Reflectance_Land',
                   'QualityWeight_Path_Radiance_Land' : __DataPath + \
                                 'QualityWeight_Path_Radiance_Land',
                   'QualityWeight_Critical_Reflectance_Land' : __DataPath + \
                                 'QualityWeight_Critical_Reflectance_Land',
                   'Aerosol_Type_Land' : __DataPath+'Aerosol_Type_Land',
                   'Fitting_Error_Land' : __DataPath+'Fitting_Error_Land',
                   'Surface_Reflectance_Land' : __DataPath + \
                                 'Surface_Reflectance_Land',
                   'Corrected_Optical_Depth_Land' : __DataPath + \
                                 'Corrected_Optical_Depth_Land',
                   'Corrected_Optical_Depth_Land_wav2p1' : __DataPath + \
                                 'Corrected_Optical_Depth_Land_wav2p1',
                   'Optical_Depth_Small_Land' : __DataPath + \
                                 'Optical_Depth_Small_Land',
                   'Optical_Depth_Ratio_Small_Land' : __DataPath + \
                                 'Optical_Depth_Ratio_Small_Land',
                   'Number_Pixels_Used_Land' : __DataPath + \
                                 'Number_Pixels_Used_Land',
                   'Mean_Reflectance_Land' : __DataPath+'Mean_Reflectance_Land',
                   'STD_Reflectance_Land' : __DataPath+'STD_Reflectance_Land',
                   'Mass_Concentration_Land' : __DataPath + \
                                 'Mass_Concentration_Land',
                   'Angstrom_Exponent_Land' : __DataPath + \
                                 'Anstrom_Exponent_Land',
                   'Cloud_Fraction_Land' : __DataPath+'Cloud_Fraction_Land',
                   'Quality_Assurance_Land' : __DataPath + \
                                 'Quality_Assurance_Land',
                   'Quality_Assurance_Crit_Ref_Land' : __DataPath + \
                                 'Quality_Assurance_Crit_Ref_Land',
                   'Aerosol_Cldmask_Byproducts_Land' : __DataPath + \
                                 'Aerosol_Cldmask_Byproducts_Land',
                   'Solution_Index_Ocean_Small' : __DataPath + \
                                 'Solution_Index_Ocean_Small',
                   'Solution_Index_Ocean_Large' : __DataPath + \
                                 'Solution_Index_Ocean_Large',
                   'Effective_Optical_Depth_Best_Ocean' : __DataPath + \
                                 'Effective_Optical_Depth_Best_Ocean',
                   'Effective_Optical_Depth_Average_Ocean' : __DataPath + \
                                 'Effective_Optical_Depth_Average_Ocean',
                   'Optical_Depth_Small_Best_Ocean' : __DataPath + \
                                 'Optical_Depth_Small_Best_Ocean',
                   'Optical_Depth_Small_Average_Ocean' : __DataPath + \
                                 'Optical_Depth_Small_Average_Ocean',
                   'Optical_Depth_Large_Best_Ocean' : __DataPath + \
                                 'Optical_Depth_Large_Best_Ocean',
                   'Optical_Depth_Large_Average_Ocean' : __DataPath + \
                                 'Optical_Depth_Large_Average_Ocean',
                   'Mass_Concentration_Ocean' : __DataPath + \
                                 'Mass_Concentration_Ocean',
                   'Effective_Radius_Ocean' : __DataPath + \
                                 'Effective_Radius_Ocean',
                   'Cloud_Condensation_Nuclei_Ocean' : __DataPath + \
                                 'Cloud_Condensation_Nuclei_Ocean',
                   'Asymmetry_Factor_Best_Ocean' : __DataPath + \
                                 'Asymmetry_Factor_Best_Ocean',
                   'Asymmetry_Factor_Average_Ocean' : __DataPath + \
                                 'Assymetry_Factor_Average_Ocean',
                   'Backscattering_Ratio_Best_Ocean' : __DataPath + \
                                 'Backscattering_Ratio_Best_Ocean',
                   'Backscattering_Ratio_Average_Ocean' : __DataPath + \
                                 'Backscattering_Ratio_Average_Ocean',
                   'Angstrom_Exponent_1_Ocean' : __DataPath + \
                                 'Angstrom_Exponent_1_Ocean',
                   'Angstrom_Exponent_2_Ocean' : __DataPath + \
                                 'Angstrom_Exponent_2_Ocean',
                   'Least_Squares_Error_Ocean' : __DataPath + \
                                 'Least_Squares_Error_Ocean',
                   'Optical_Depth_Ratio_Small_Ocean_0.55micron' : __DataPath + \
                                 'Optical_Depth_Ratio_Small_Ocean_0.55micron',
                   'Optical_Depth_by_models_ocean' : __DataPath + \
                                 'Optical_Depth_by_models_ocean',
                   'Cloud_Fraction_Ocean' : __DataPath+'Cloud_Fraction_Ocean',
                   'Number_Pixels_Used_Ocean' : __DataPath + \
                                 'Number_Pixels_Used_Ocean',
                   'Mean_Reflectance_Ocean' : __DataPath + \
                                 'Mean_Reflectance_Ocean',
                   'STD_Reflectance_Ocean' : __DataPath + \
                                 'STD_Reflectance_Ocean',
                   'Quality_Assurance_Ocean' : __DataPath + \
                                 'Quality_Assurance_Ocean',
                   'Aerosol_Cldmask_Byproducts_Ocean' : __DataPath + \
                                 'Aerosol_Cldmask_Byproducts_Ocean',
                   'Deep_Blue_Aerosol_Optical_Depth_550_Land' : __DataPath + \
                                 'Deep_Blue_Aerosol_Optical_Depth_550_Land',
                   'Deep_Blue_Aerosol_Optical_Depth_Land' : __DataPath + \
                                 'Deep_Blue_Optical_Depth_Land',
                   'Deep_Blue_Angstrom_Exponent_Land' : __DataPath + \
                                 'Deep_Blue_Angstrom_Exponent_Land',
                   'Deep_Blue_Single_Scattering_Albedo_Land' : __DataPath + \
                                 'Deep_blue_Single_Scattering_Albedo_Land',
                   'Deep_Blue_Surface_Reflectance_Land' : __DataPath + \
                                 'Deep_Blue_Surface_Reflectance_Land',
                   'Deep_Blue_Mean_Reflectance_Land' : __DataPath + \
                                 'Deep_Blue_Mean_Reflectance_Land',
                   'Deep_Blue_Number_Pixels_Used_Land' : __DataPath + \
                                 'Deep_Blue_Number_Pixels_Used_Land',
                   'Deep_Blue_Aerosol_Optical_Depth_550_Land_STD' : __DataPath+\
                                 'Deep_Blue_Aerosol_Optical_Depth_550_Land_STD',
                   'Deep_Blue_Aerosol_Optical_Depth_Land_STD' : __DataPath + \
                                 'Deep_Blue_Aerosol_Optical_Depth_Land_Std'}

    _indexMap = {'default' : lambda var, ind: var[..., ind[0], ind[1]],
                 'Quality_Assurance_Land' : lambda var, ind: var[..., ind[0], ind[1], 0],
                 'Quality_Assurance_Ocean' : lambda var, ind: var[..., ind[0], ind[1], 0]}

    _FVMap = {'Latitude' : -999.0,
              'Longitude' : -999.0,
              'Scan_Start_Time' : -999.0,
              'Solar_Zenith' : -9999,
              'Solar_Azimuth' : -9999,
              'Sensor_Zenith' : -9999,
              'Sensor_Azimuth' : -9999,
              'Cloud_Mask_QA' : 0,
              'Scattering_Angle' : -9999,
              'Optical_Depth_Land_And_Ocean' : -9999,
              'Image_Optical_Depth_Land_And_Ocean' : -9999,
              'Optical_Depth_Ratio_Small_Land_And_Ocean' : -9999,
              'Mean_Reflectance_Land_All' : -9999,
              'Standard_Deviation_Reflectance_Land_All' : -9999,
              'Path_Radiance_Land' : -9999,
              'Error_Path_Radiance_Land' : -9999,
              'Critical_Reflectance_Land' : -9999,
              'Error_Critical_Reflectance_Land' : -9999,
              'QualityWeight_Path_Radiance_Land' : -9999,
              'QualityWeight_Critical_Reflectance_Land' : -9999, 
              'Aerosol_Type_Land' : -9999,
              'Fitting_Error_Land' : -9999,
              'Surface_Reflectance_Land' : -9999,
              'Corrected_Optical_Depth_Land' : -9999,
              'Corrected_Optical_Depth_Land_wav2p1' : -9999,
              'Optical_Depth_Small_Land' : -9999,
              'Optical_Depth_Ratio_Small_Land' : -9999,
              'Number_Pixels_Used_Land' : -9999,
              'Mean_Reflectance_Land' : -9999,
              'STD_Reflectance_Land' : -9999,
              'Mass_Concentration_Land' : -999.0,
              'Angstrom_Exponent_Land' : -9999,
              'Cloud_Fraction_Land' : -9999,
              'Quality_Assurance_Land' : 0,
              'Quality_Assurance_Crit_Ref_Land' : 0,
              'Aerosol_Cldmask_Byproducts_Land' : -9999,
              'Solution_Index_Ocean_Small' : -9999,
              'Solution_Index_Ocean_Large' : -9999,
              'Effective_Optical_Depth_Best_Ocean' : -9999,
              'Effective_Optical_Depth_Average_Ocean' : -9999,
              'Optical_Depth_Small_Best_Ocean' : -9999,
              'Optical_Depth_Small_Average_Ocean' : -9999,
              'Optical_Depth_Large_Best_Ocean' : -9999,
              'Optical_Depth_Large_Average_Ocean' : -9999,
              'Mass_Concentration_Ocean' : -999.0,
              'Effective_Radius_Ocean' : -9999,
              'Cloud_Condensation_Nuclei_Ocean' : -999.0,
              'Asymmetry_Factor_Best_Ocean' : -9999,
              'Asymmetry_Factor_Average_Ocean' : -9999,
              'Backscattering_Ratio_Best_Ocean' : -9999,
              'Backscattering_Ratio_Average_Ocean' : -9999,
              'Angstrom_Exponent_1_Ocean' : -9999,
              'Angstrom_Exponent_2_Ocean' : -9999,
              'Least_Squares_Error_Ocean' : -9999,
              'Optical_Depth_Ratio_Small_Ocean_0.55micron' : -9999,
              'Optical_Depth_by_models_ocean' : -9999,
              'Cloud_Fraction_Ocean' : -9999,
              'Number_Pixels_Used_Ocean' : -9999,
              'Mean_Reflectance_Ocean' : -9999,
              'STD_Reflectance_Ocean' : -9999,
              'Quality_Assurance_Ocean' : 0,
              'Aerosol_Cldmask_Byproducts_Ocean' : -9999,
              'Deep_Blue_Aerosol_Optical_Depth_550_Land' : -9999,
              'Deep_Blue_Aerosol_Optical_Depth_Land' : -9999,
              'Deep_Blue_Angstrom_Exponent_Land' : -9999,
              'Deep_Blue_Single_Scattering_Albedo_Land' : -9999,
              'Deep_Blue_Surface_Reflectance_Land' : -9999,
              'Deep_Blue_Mean_Reflectance_Land' : -9999,
              'Deep_Blue_Number_Pixels_Used_Land' : -9999,
              'Deep_Blue_Aerosol_Optical_Depth_550_Land_STD' : -9999,
              'Deep_Blue_Aerosol_Optical_Depth_Land_STD' : -9999}

    _ScaleOffsetMap = {'Latitude' : None,
                   'Longitude' : None,
                   'Scan_Start_Time' : None,
                   'Solar_Zenith' : (0.009999999776482582,0.0),
                   'Solar_Azimuth' : (0.009999999776482582,0.0),
                   'Sensor_Zenith' : (0.009999999776482582,0.0),
                   'Sensor_Azimuth' : (0.009999999776482582,0.0),
                   'Cloud_Mask_QA' : None,
                   'Scattering_Angle' : (0.009999999776482582,0.0),
                   'Optical_Depth_Land_And_Ocean' : (0.0010000000474974513,0.0),
                   'Image_Optical_Depth_Land_And_Ocean' : (0.0010000000474974513,0.0),
                   'Optical_Depth_Ratio_Small_Land_And_Ocean' : (0.0010000000474974513,0.0),
                   'Mean_Reflectance_Land_All' : (0.00009999999747378752, 0.0),
                   'Standard_Deviation_Reflectance_Land_All' : (0.00009999999747378752,0.0),
                   'Path_Radiance_Land' : (0.00009999999747378752, 0.0),
                   'Error_Path_Radiance_Land' : (0.0, 0.00009999999747378752),
                   'Critical_Reflectance_Land' : (0.00009999999747378752, 0.0),
                   'Error_Critical_Reflectance_Land' : (0.00009999999747378752, 0.0),
                   'QualityWeight_Path_Radiance_Land' : None,
                   'QualityWeight_Critical_Reflectance_Land' : None, 
                   'Aerosol_Type_Land' : None,
                   'Fitting_Error_Land' : (0.0010000000474974513, 0.0),
                   'Surface_Reflectance_Land' : (0.0010000000474974513, 0.0),
                   'Corrected_Optical_Depth_Land' : (0.0010000000474974513, 0.0),
                   'Corrected_Optical_Depth_Land_wav2p1' : (0.0010000000474974513,0.0),
                   'Optical_Depth_Small_Land' : (0.0010000000474974513,0.0),
                   'Optical_Depth_Ratio_Small_Land' : (0.0010000000474974513, 0.0),
                   'Number_Pixels_Used_Land' : None,
                   'Mean_Reflectance_Land' : (0.00009999999747378752, 0.0),
                   'STD_Reflectance_Land' : (0.00009999999747378752, 0.0),
                   'Mass_Concentration_Land' : None,
                   'Angstrom_Exponent_Land' : (0.0010000000474974513, 0.0),
                   'Cloud_Fraction_Land' : (0.0010000000474974513, 0.0),
                   'Quality_Assurance_Land' : None,
                   'Quality_Assurance_Crit_Ref_Land' : None,
                   'Aerosol_Cldmask_Byproducts_Land' : (0.0010000000474974513, 0.0),
                   'Solution_Index_Ocean_Small' : None,
                   'Solution_Index_Ocean_Large' : None,
                   'Effective_Optical_Depth_Best_Ocean' : (0.0010000000474974513, 0.0),
                   'Effective_Optical_Depth_Average_Ocean' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_Small_Best_Ocean' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_Small_Average_Ocean' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_Large_Best_Ocean' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_Large_Average_Ocean' : (0.0010000000474974513, 0.0),
                   'Mass_Concentration_Ocean' : None,
                   'Effective_Radius_Ocean' : (0.0010000000474974513, 0.0),
                   'Cloud_Condensation_Nuclei_Ocean' : None,
                   'Asymmetry_Factor_Best_Ocean' : (0.0010000000474974513, 0.0),
                   'Asymmetry_Factor_Average_Ocean' : (0.0010000000474974513, 0.0),
                   'Backscattering_Ratio_Best_Ocean' : (0.0010000000474974513, 0.0),
                   'Backscattering_Ratio_Average_Ocean' : (0.0010000000474974513, 0.0),
                   'Angstrom_Exponent_1_Ocean' : (0.0010000000474974513, 0.0),
                   'Angstrom_Exponent_2_Ocean' : (0.0010000000474974513, 0.0),
                   'Least_Squares_Error_Ocean' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_Ratio_Small_Ocean_0.55micron' : (0.0010000000474974513, 0.0),
                   'Optical_Depth_by_models_ocean' : (0.0010000000474974513, 0.0),
                   'Cloud_Fraction_Ocean' : (0.0010000000474974513, 0.0),
                   'Number_Pixels_Used_Ocean' : None,
                   'Mean_Reflectance_Ocean' : (0.00009999999747378752, 0.0),
                   'STD_Reflectance_Ocean' : (0.00009999999747378752, 0.0),
                   'Quality_Assurance_Ocean' : None,
                   'Aerosol_Cldmask_Byproducts_Ocean' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Aerosol_Optical_Depth_550_Land' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Aerosol_Optical_Depth_Land' :(0.0010000000474974513, 0.0),
                   'Deep_Blue_Angstrom_Exponent_Land' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Single_Scattering_Albedo_Land' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Surface_Reflectance_Land' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Mean_Reflectance_Land' : (0.00009999999747378752, 0.0),
                   'Deep_Blue_Number_Pixels_Used_Land' : None,
                   'Deep_Blue_Aerosol_Optical_Depth_550_Land_STD' : (0.0010000000474974513, 0.0),
                   'Deep_Blue_Aerosol_Optical_Depth_Land_STD' : (0.0010000000474974513, 0.0)}

    def get(self, key, indices=None):
        '''Overloaded version of get that applies the correct missing value.'''
        if self._ScaleOffsetMap[key]:
            retval = HDF4File.get(self, key, indices, missingValue=self._FVMap[key])*self._ScaleOffsetMap[key][0]+self._ScaleOffsetMap[key][1]
            return retval                                              
        else:
            retval = HDF4File.get(self, key, indices, missingValue=self._FVMap[key])
            return retval

    def get_cm(self, key, indices=None):
        '''Overloaded version of get_cm that applied the correct missing value.'''
        if self._ScaleOffsetMap[key]:
           return HDF4File.get(self, key, indices, missingValue=self._FVMap[key])*self._ScaleOffsetMap[key][0]+self._ScaleOffsetMap[key][1]
        else:
        #    print("in parse_geo.py get_cm function with HDF4 class")
        #    print(key)
        #    print(indices)
           return HDF4File.get(self, key, indices, missingValue=self._FVMap[key])
        
    def get_geo_centers(self):
        lat = self.get('Latitude').squeeze()
        lon = self.get('Longitude').squeeze()
        tem = numpy.arange(lat.size).reshape(lat.shape)
        ind = numpy.transpose(numpy.array([tem/tem[0].size, tem%tem[0].size]),\
                              (1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), \
                              ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape, dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

    def get_geo_corners(self):
        '''Regional intersect cannot be used with MODIS data'''
        raise NotImplementedError(\
                                  'The regional_intersect and '\
                                  'global_intersect\n                     '\
                                  'mapping functions cannot be used with '\
                                  'MODIS data,\n                     '\
                                  'as no corner information is provided by '\
                                  'this filetype.\n                     Please'\
                                  ' use the point_in_cell mapping function '\
                                  'instead.')

class HDFnasaomihchol2_File(HDFFile):
    """
    Provide interface to NASA OMI HCHO L2 product, with pixel corners
    
    Pixel corners are retrieved from an extra file that must be accessible
    in cornerDir.  The files listed in cornerFileList, if any, will be checked
    first, followed by any files in cornerDir. If no valid pixel corner file
    is found that matches the orbit number of the input file, the parser will
    instantiage but get_geo_corners will fail with an IOError. 

    The corners retrieved are for the visible channel
    used by the NO2 algorithm- using this parser for other products may 
    require altering the parser to use a different channel if 
    appropriate.
    
    The keys to retrieve variables are the names of the variables within actual
    files.  Note that the NASA product documentation has the wrong name for 
    the "SlantColumnAmountH20Std" variable due to a case typo.

    Does not support the fields "UnpolFldCoefficients" or "SmallPixelRadiance"
    because the dimensionality of these fields changes between files and there
    is no way to deal with fields with variable dimension size in the current 
    framework.
    """
    
    OMIAURAHCHO_FILE_NAME = "OMI/Aura Formaldehyde (HCHO) Total "\
                           "Column 1-Orbit L2 Swath 13x24km"
    OMIAURAHCHO_CORNER_FILE_NAME = "OMI/Aura Global Ground Pixel Corners "\
                                  "1-Orbit L2 Swath 13x24km"

    def __init__(self, filename, subtype='', extension=None, cornerDir=None,
                 cornerFileList=None):
        HDFFile.__init__(self, filename, subtype, extension)

        # make sure filename is actually an input file
        if getLongName(filename) != HDFnasaomihchol2_File.OMIAURAHCHO_FILE_NAME:
            raise IOError('Attempt to read non-NASA OMI L2 file as such.')
        
        # start by assuming we aren't going to find anything
        self.pixCorners = None

        # see if the corner directory even exists.  If it doesn't, we obviously can't 
        # find a corner file
        if os.path.isdir(cornerDir):

            # convert the corner files into full pathnames
            # unless we were given null string (signal to search directory)
            if cornerFileList != ['']:
                cornerFileList = [os.path.join(cornerDir, f) for f in cornerFileList]

            # get orbit number of file for matching
            forbitnumber = getOrbitNumber(filename)

            # try using the list
        
            for f in cornerFileList:
                if f != '' and getLongName(f) == HDFnasaomihchol2_File.OMIAURAHCHO_CORNER_FILE_NAME:
                    try:
                        if getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass

            # if necessary, search entire corner file directory
            if self.pixCorners == None:        
                allPossible = [os.path.join(cornerDir, f) for f in os.listdir(cornerDir)]
                for f in allPossible:
                    try:
                        if tables.is_hdf5_file(f) \
                                and getLongName(f) == HDFnasaomihchol2_File.OMIAURAHCHO_CORNER_FILE_NAME \
                                and getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass
                
        if self.pixCorners == None:
            print("No valid corner file found for {0}.".format(filename))                        
            
    __dataPath = '/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
    __geoPath = '/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'
    
    _nameExpMap = {'GroundPixelQualityFlags' : __dataPath+'GroundPixelQualityFlags',
                   'ColumnAmount' : __dataPath+'ColumnAmount',
                   'ColumnAmountDestriped' :__dataPath+'ColumnAmountDestriped',
                   'ColumnAmountUncertainty' :__dataPath+'ColumnAmountUncertainty' ,
                   'ClimatologyLevels' :__dataPath+'ClimatologyLevels',
                   'ColumnAmount' :__dataPath+'ColumnAmount',
                   'ColumnAmountDestriped' :__dataPath+'ColumnAmountDestriped',
                   'ColumnUncertainty' :__dataPath+'ColumnUncertainty',
                   'GasProfile' :__dataPath+'GasProfile',
                   'RadianceReferenceColumnAmount' :__dataPath+'RadianceReferenceColumnAmount',
                   'RadianceReferenceColumnUncertainty' :__dataPath + 'RadianceReferenceColumnUncertainty',
                   'ReferenceSectorCorrectedVerticalColumn' :__dataPath+'ReferenceSectorCorrectedVerticalColumn',
                   'ScatteringWeights' :__dataPath+'ScatteringWeights',
                   'SolarWavCalConvergenceFlag' :__dataPath+'SolarWavCalConvergenceFlag',
                   'AMFCloudFraction' :__dataPath+'AMFCloudFraction',
                   'AMFCloudPressure' :__dataPath+'AMFCloudPressure',
                   'AirMassFactor' :__dataPath+'AirMassFactor',
                   'AirMassFactorDiagnosticFlag' :__dataPath+'AirMassFactorDiagnosticFlag',
                   'AirMassFactorGeometric' :__dataPath+'AirMassFactorGeometric',
                   'Albedo' :__dataPath+'Albedo',
                   'PixelCornerLongitudes' :__dataPath+'PixelCornerLongitudes',
                   'PixelCornerLatitudes' :__dataPath+'PixelCornerLatitudes',                   	
                   'MainDataQualityFlag' :__dataPath+'MainDataQualityFlag',
                   'FitConvergenceFlag' : __dataPath+'FitConvergenceFlag',
                   'RadianceReferenceConvergenceFlag' : __dataPath+'RadianceReferenceConvergenceFlag',
                   'RadianceWavCalConvergeFlag' : __dataPath+'RadianceWavCalConvergeFlag',
                   'XtrackQualityFlags' : __geoPath+'XtrackQualityFlags',
                   'Latitude' : __geoPath+'Latitude',
                   'Longitude' : __geoPath+'Longitude',
                   'TerrainHeight' :__geoPath+'TerrainHeight',
                   'SolarAzimuthAngle' : __geoPath+'SolarAzimuthAngle',
                   'SolarZenithAngle' : __geoPath+'SolarZenithAngle',
                   'SpacecraftAltitude' : __geoPath+'SpacecraftAltitude',
                   'Time' : __geoPath+'Time',
                   'ViewingAzimuthAngle' : __geoPath+'ViewingAzimuthAngle',
                   'ViewingZenithAngle' : __geoPath+'ViewingZenithAngle'}
    
    _indexMap = {'default' : lambda var, ind: var[..., ind[0], ind[1]],
    	          'Time' : lambda var, ind: var[ind[0]]}
                  	    
    
    def get_geo_corners(self):
        '''
        Retrieves array of the corners of the pixels.  
        
        Throws IOError if no pixel corner file specified
        '''
        latNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLatitude'
        lonNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLongitude'
        try:
            pxFid = tables.open_file(self.pixCorners)
        except AttributeError:
            raise IOError('Unable to open pixel corners file.  Need pixel corners file to use corners')
        try:
            latNode = pxFid.get_node('/', latNodeName)
            lonNode = pxFid.get_node('/', lonNodeName)
            # Note: it is assumed that there are no missing values.
            lat = latNode[:].transpose((1,2,0))
            lon = lonNode[:].transpose((1,2,0))            
        finally:
            pxFid.close()
        ind = numpy.indices(lat.shape[0:2]).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
    
    def get_geo_centers(self):
        lat = self.get('Latitude')
        lon = self.get('Longitude')
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape, dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

class HDFnasaomiv3l2_File(HDFFile):
    """
    Provide interface to NASA OMI L2 product, with pixel corners
    
    Pixel corners are retrieved from an extra file that must be accessible
    in cornerDir.  The files listed in cornerFileList, if any, will be checked
    first, followed by any files in cornerDir. If no valid pixel corner file
    is found that matches the orbit number of the input file, the parser will
    instantiage but get_geo_corners will fail with an IOError. 

    The corners retrieved are for the visible channel
    used by the NO2 algorithm- using this parser for other products may 
    require altering the parser to use a different channel if 
    appropriate.
    
    The keys to retrieve variables are the names of the variables within actual
    files.  Note that the NASA product documentation has the wrong name for 
    the "SlantColumnAmountH20Std" variable due to a case typo.

    Does not support the fields "UnpolFldCoefficients" or "SmallPixelRadiance"
    because the dimensionality of these fields changes between files and there
    is no way to deal with fields with variable dimension size in the current 
    framework.
    """
    
    OMIAURANO2_FILE_NAME = "OMI/Aura Nitrogen Dioxide (NO2) Total & "\
                           "Troposph.Column 1-Orbit L2 Swath 13x24km"
    OMIAURANO2_CORNER_FILE_NAME = "OMI/Aura Global Ground Pixel Corners "\
                                  "1-Orbit L2 Swath 13x24km"

    def __init__(self, filename, subtype='', extension=None, cornerDir=None,
                 cornerFileList=None):
        HDFFile.__init__(self, filename, subtype, extension)

        # make sure filename is actually an input file
        if getLongName(filename) != HDFnasaomiv3l2_File.OMIAURANO2_FILE_NAME:
            raise IOError('Attempt to read non-NASA OMI L2 file as such.')
       
        # start by assuming we aren't going to find anything
        self.pixCorners = None

        # see if the corner directory even exists.  If it doesn't, we obviously can't 
        # find a corner file
        if os.path.isdir(cornerDir):

            # convert the corner files into full pathnames
            # unless we were given null string (signal to search directory)
            if cornerFileList != ['']:
                cornerFileList = [os.path.join(cornerDir, f) for f in cornerFileList]

            # get orbit number of file for matching
            forbitnumber = getOrbitNumber(filename)

            # try using the list
        
            for f in cornerFileList:
                if f != '' and getLongName(f) == HDFnasaomiv3l2_File.OMIAURANO2_CORNER_FILE_NAME:
                    try:
                        if getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass

            # if necessary, search entire corner file directory
            if self.pixCorners == None:        
                allPossible = [os.path.join(cornerDir, f) for f in os.listdir(cornerDir)]
                for f in allPossible:
                    try:
                        if tables.is_hdf5_file(f) \
                                and getLongName(f) == HDFnasaomil2_File.OMIAURANO2_CORNER_FILE_NAME \
                                and getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass
                
        if self.pixCorners == None:
            print("No valid corner file found for {0}.".format(filename))              
    __dataPath = '/HDFEOS/SWATHS/ColumnAmountNO2/Data Fields/'
    __geoPath = '/HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields/'
    
    _nameExpMap = {'Time' : __geoPath+'Time',
                 'Latitude' : __geoPath+'Latitude',
                 'Longitude' :__geoPath+'Longitude',
                 'SpacecraftLatitude' : __geoPath+'SpacecraftLatitude', 
                 'SpacecraftLongitude' : __geoPath+'SpacecraftLongitude',
                 'SpacecraftAltitude' : __geoPath+'SpacecraftAltitude',
                 'SolarZenithAngle' : __geoPath+'SolarZenithAngle',
                 'SolarAzimuthAngle' : __geoPath+'SolarAzimuthAngle',
                 'ViewingZenithAngle' : __geoPath+'ViewingZenithAngle',
                 'ViewingAzimuthAngle' : __geoPath+'ViewingAzimuthAngle',
                 'ColumnAmountNO2' : __dataPath+'ColumnAmountNO2',
                 'ColumnAmountNO2Std' : __dataPath+'ColumnAmountNO2Std',
                 'ColumnAmountNO2Strat' : __dataPath+'ColumnAmountNO2Strat' ,
                 'ColumnAmountNO2StratStd' : __dataPath+'ColumnAmountNO2StratStd',
                 'ColumnAmountNO2Trop' : __dataPath+'ColumnAmountNO2Trop',
                 'ColumnAmountNO2TropStd' : __dataPath+'ColumnAmountNO2TropStd',
                 'SlantColumnAmountNO2' : __dataPath+'SlantColumnAmountNO2',
                 'SlantColumnAmountNO2Std' : __dataPath+'SlantColumnAmountNO2Std',
                 'SlantColumnAmountNO2Destriped' : __dataPath+'SlantColumnAmountNO2Std',
                 'RingCoefficient' : __dataPath+'RingCoefficient',
                 'RingCoefficientStd' : __dataPath+'RingCoefficientStd',
                 'SlantColumnAmountO3' : __dataPath+'SlantColumnAmountO3',
                 'SlantColumnAmountO3Std' : __dataPath+'SlantColumnAmountO3Std',
                 'SlantColumnAmountH2O' : __dataPath+'SlantColumnAmountH2O',
                 'SlantColumnAmountH2Otd' : __dataPath+'SlantColumnAmountH2Otd',
                 'SlantColumnAmountO2O2' : __dataPath+'SlantColumnAmountO2O2',
                 'SlantcolumnAmountO2O2Std' : __dataPath+'SlantcolumnAmountO2O2Std',
                 'PolynomialCoefficients' : __dataPath +'PolynomialCoefficients',
                 'PolynomialCoefficientsStd' : __dataPath + 'PolynomialCoefficientsStd',
                 'ChiSquaredOfFit' : __dataPath+'ChiSquaredOfFit',
                 'RootMeanSquareErrorOfFit' : __dataPath+'RootMeanSquareErrorOfFit',
                 'AmfQualityFlags' : __dataPath+'AmfQualityFlags',
                 'AlgorithmFlags' : __dataPath+'AlgorithmFlags',
                 'AmfStrat' : __dataPath+'AmfStrat',
                 'AmfStratClear' : __dataPath+'AmfStratClear',
                 'AmfStratCloudy' : __dataPath+'AmfStratCloudy',
                 'AmfStratStd' : __dataPath+ 'AmfStratStd',
                 'AmfTrop' : __dataPath+'AmfTrop',
                 'AmfTropClear' : __dataPath+'AmfTropClear',
                 'AmfTropCloudy' : __dataPath+'AmfTropCloudy',
                 'AmfTropStd' : __dataPath+'AmfTropStd',
                 'CloudFraction' : __dataPath+'CloudFraction' ,
                 'CloudFractionStd' : __dataPath+'CloudFractionStd',
                 'CloudRadianceFraction' : __dataPath+'CloudRadianceFraction',
                 'CloudPressure' : __dataPath+'CloudPressure',
                 'CloudPressureStd' : __dataPath+'CloudPressureStd',
                 'TerrainReflectivity' : __dataPath+'TerrainReflectivity',
                 'TerrainPressure' : __dataPath+'TerrainPressure',
                 'TerrainHeight' : __dataPath+'TerrainHeight' ,
                 'TropopausePressure' : __dataPath+'TropopausePressure',
                 'VcdApTrop' : __dataPath+'VcdApTrop',
                 'SlantColumnAmountNO2' : __dataPath+'SlantColumnAmountNO2',
                 'SlantColumnAmountNO2Destriped' : __dataPath+'SlantColumnAmountNO2Destriped',
                 'SmallPixelRadiancePointer' :__dataPath +'SmallPixelRadiancePointer',
                 'SmallPixelRadiance' : __dataPath+'SmallPixelRadiance',
                 'InstrumentConfigurationId' : __dataPath+'InstrumentConfigurationId',
                 'MeasurementQualityFlags' : __dataPath+'MeasurementQualityFlags',
                 'FitQualityFlags' : __dataPath+'FitQualityFlags',
                 'WavelengthRegistrationCheck' : __dataPath+ 'WavelengthRegistrationCheck',
                 'WavelengthRegistrationCheckStd' : __dataPath+'UnpolFldLatBandQualityFlags',
                 'UnpolFldLatBandQualityFlags' : __dataPath+'UnpolFldLatBandQualityFlags',
                 'VcdQualityFlags' : __dataPath+'VcdQualityFlags',
                 'XTrackQualityFlags' : __dataPath+'XTrackQualityFlags',
                 'ScatteringWeight' : __dataPath+ 'ScatteringWeight', 
                 'ScatteringWtPressure' : __dataPath+'ScatteringWtPressure',
                 'FoV75CornerLatitude' :__geoPath+'FoV75CornerLatitude',
                 'FoV75CornerLongitude' :__geoPath+'FoV75CornerLongitude'}
                 
    
    _indexMap = {'default' : lambda var, ind: var[ind[0], ind[1], ...],
                  'SmallPixelRadiance' : lambda var, ind: var[:, ind[1]],
                  'SmallPixelRadiancePointer' : lambda var, ind: var[ind[0], :],
                  'InstrumentConfigurationId' : lambda var, ind: var[ind[0]],
                  'MeasurementQualityFlags' : lambda var, ind: var[ind[0]],
                  'WavelengthRegistrationCheck' : lambda var, ind: var[ind[0], :],
                  'WavelengthRegistrationCheckStd' : lambda var, ind: var[ind[0], :],
                  'UnpolFldLatBandQualityFlags' : lambda var, ind: var[:],
                  'Time' : lambda var, ind: var[ind[0]],
                  'SpacecraftLatitude' : lambda var, ind: var[ind[0]],
                  'SpacecraftLongitude' : lambda var, ind: var[ind[0]],
                  'SpacecraftAltitude' : lambda var, ind: var[ind[0]],
                  'ScatteringWtPressure' : lambda var, ind: var[:]}
    
    def get_geo_corners(self):
        '''
        Retrieves array of the corners of the pixels.  
        
        Throws IOError if no pixel corner file specified
        '''
        latNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLatitude'
        lonNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLongitude'
        try:
            pxFid = tables.open_file(self.pixCorners)
        except AttributeError:
            raise IOError('Unable to open pixel corners file.  Need pixel corners file to use corners')
        try:
            latNode = pxFid.get_node('/', latNodeName)
            lonNode = pxFid.get_node('/', lonNodeName)
            # Note: it is assumed that there are no missing values.
            lat = latNode[:].transpose((1,2,0))
            lon = lonNode[:].transpose((1,2,0))            
        finally:
            pxFid.close()
        ind = numpy.indices(lat.shape[0:2]).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
    
    def get_geo_centers(self):
        lat = self.get('Latitude')
        lon = self.get('Longitude')
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape, dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)

class HDFnasaomiso2l2_File(HDFFile):
    """
    Provide interface to NASA OMI SO2 L2 product, with pixel corners
    
    Pixel corners are retrieved from an extra file that must be accessible
    in cornerDir.  The files listed in cornerFileList, if any, will be checked
    first, followed by any files in cornerDir. If no valid pixel corner file
    is found that matches the orbit number of the input file, the parser will
    instantiage but get_geo_corners will fail with an IOError. 

    The corners retrieved are for the visible channel
    used by the NO2 algorithm- using this parser for other products may 
    require altering the parser to use a different channel if 
    appropriate.
    """
    
    OMIAURASO2_FILE_NAME = "OMI/Aura Sulphur Dioxide (SO2) Total "\
                           "Column 1-Orbit L2 Swath 13x24km"
    OMIAURASO2_CORNER_FILE_NAME = "OMI/Aura Global Ground Pixel Corners "\
                                  "1-Orbit L2 Swath 13x24km"

    def __init__(self, filename, subtype='', extension=None, cornerDir=None,
                 cornerFileList=None):
        HDFFile.__init__(self, filename, subtype, extension)

        # make sure filename is actually an input file
        if getLongName(filename) != HDFnasaomiso2l2_File.OMIAURASO2_FILE_NAME:
            raise IOError('Attempt to read non-NASA OMI L2 file as such.')
        
        # start by assuming we aren't going to find anything
        self.pixCorners = None

        # see if the corner directory even exists.  If it doesn't, we obviously can't 
        # find a corner file
        if os.path.isdir(cornerDir):

            # convert the corner files into full pathnames
            # unless we were given null string (signal to search directory)
            if cornerFileList != ['']:
                cornerFileList = [os.path.join(cornerDir, f) for f in cornerFileList]

            # get orbit number of file for matching
            forbitnumber = getOrbitNumber(filename)

            # try using the list
        
            for f in cornerFileList:
                if f != '' and getLongName(f) == HDFnasaomiso2l2_File.OMIAURASO2_CORNER_FILE_NAME:
                    try:
                        if getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass

            # if necessary, search entire corner file directory
            if self.pixCorners == None:        
                allPossible = [os.path.join(cornerDir, f) for f in os.listdir(cornerDir)]
                for f in allPossible:
                    try:
                        if tables.is_hdf_file(f) \
                                and getLongName(f) == HDFnasaomiso2l2_File.OMIAURASO2_CORNER_FILE_NAME \
                                and getOrbitNumber(f) == forbitnumber:
                            self.pixCorners = f
                            break
                    except:
                        pass
                
        if self.pixCorners == None:
            print("No valid corner file found for {0}.".format(filename))                        
            
    __dataPath = '/HDFEOS/SWATHS/OMI Total Column Amount SO2/Data Fields/'
    __geoPath = '/HDFEOS/SWATHS/OMI Total Column Amount SO2/Geolocation Fields/'
    
    _nameExpMap = {'AlgorithmFlag_PBL' : __dataPath+'AlgorithmFlag_PBL',
                   'AlgorithmFlag_STL' : __dataPath+'AlgorithmFlag_STL',
                   'AlgorithmFlag_TRL' : __dataPath+'AlgorithmFlag_TRL',
                   'AlgorithmFlag_TRM' : __dataPath+'AlgorithmFlag_TRM',
                   'ChiSquareLfit' : __dataPath+'ChiSquareLfit',
                   'CloudPressure' : __dataPath+'CloudPressure',
                   'ColumnAmountO3' : __dataPath+'ColumnAmountO3',
                   'ColumnAmountSO2_PBL' : __dataPath+'ColumnAmountSO2_PBL',
                   'ColumnAmountSO2_PBLbrd' : __dataPath+'ColumnAmountSO2_PBLbrd',
                   'ColumnAmountSO2_STL' : __dataPath+'ColumnAmountSO2_STL',
                   'ColumnAmountSO2_STLbrd' : __dataPath+'ColumnAmountSO2_STLbrd',
                   'ColumnAmountSO2_TRL' : __dataPath+'ColumnAmountSO2_TRL',
                   'ColumnAmountSO2_TRM' : __dataPath+'ColumnAmountSO2_TRM',
                   'ColumnAmountSO2_TRMbrd' : __dataPath+'ColumnAmountSO2_TRMbrd',
                   'LayerEfficiency' : __dataPath+'LayerEfficiency',
                   'QualityFlags_PBL' : __dataPath+'QualityFlags_PBL',
                   'QualityFlags_STL' : __dataPath+'QualityFlags_STL',
                   'QualityFlags_TRL' : __dataPath+'QualityFlags_TRL',
                   'QualityFlags_TRM' : __dataPath+'QualityFlags_TRM',
                   'RadiativeCloudFraction' : __dataPath+'RadiativeCloudFraction',
                   'Reflectivity331' : __dataPath+'Reflectivity331',
                   'Residual' : __dataPath+'Residual',
                   'ResidualAdjustment' : __dataPath+'ResidualAdjustment',
                   'Rlambda1st' : __dataPath+'Rlambda1st',
                   'Rlambda2nd' : __dataPath+'Rlambda2nd',
                   'SO2indexP1' : __dataPath+'SO2indexP1',
                   'SO2indexP2' : __dataPath+'SO2indexP2',
                   'SO2indexP3' : __dataPath+'SO2indexP3',
                   'TerrainPressure' : __dataPath+'TerrainPressure',
                   'UVAerosolIndex' : __dataPath+'UVAerosolIndex',
                   'Wavelength' : __dataPath+'Wavelength',
                   'dN_dSO2_STL' : __dataPath+'dN_dSO2_STL',
                   'dN_dSO2_TRL' : __dataPath+'dN_dSO2_TRL',
                   'dN_dSO2_TRM' : __dataPath+'dN_dSO2_TRM',
                   'deltaO3' : __dataPath+'deltaO3',
                   'deltaRefl' : __dataPath+'deltaRefl',
                   'fc' : __dataPath+'fc',
                   'GroundPixelQualityFlags' : __geoPath+'GroundPixelQualityFlags',
                   'Latitude' : __geoPath+'Latitude',
                   'Longitude' : __geoPath+'Longitude',
                   'RelativeAzimuthAngle' : __geoPath+'RelativeAzimuthAngle',
                   'SecondsInDay' : __geoPath+'SecondsInDay',
                   'SolarAzimuthAngle' : __geoPath+'SolarAzimuthAngle',
                   'SolarZenithAngle' : __geoPath+'SolarZenithAngle',
                   'SpacecraftAltitude' : __geoPath+'SpacecraftAltitude',
                   'SpacecraftLatitude' : __geoPath+'SpacecraftLatitude',
                   'SpacecraftLongitude' : __geoPath+'SpacecraftLongitude',
                   'TerrainHeight' : __geoPath+'TerrainHeight',
                   'Time' : __geoPath+'Time',
                   'ViewingAzimuthAngle' : __geoPath+'ViewingAzimuthAngle',
                   'ViewingZenithAngle' : __geoPath+'ViewingZenithAngle',
                   'nTimes_idx' : __geoPath+'nTimes_idx',
                   'nXtrack_idx' : __geoPath+'nXtrack_idx',
                   'FoV75CornerLatitude' :__geoPath+'FoV75CornerLatitude',
                   'FoV75CornerLongitude' :__geoPath+'FoV75CornerLongitude'}

    _indexMap = {'default' : lambda var, ind: var[ind[0], ind[1], ...],
    	          'Time' : lambda var, ind: var[ind[0]]}
                  	    
    def get_geo_corners(self):
        '''
        Retrieves array of the corners of the pixels.  
        
        Throws IOError if no pixel corner file specified
        '''
        latNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLatitude'
        lonNodeName = '/HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLongitude'
        try:
            pxFid = tables.open_file(self.pixCorners)
        except AttributeError:
            raise IOError('Unable to open pixel corners file.  Need pixel corners file to use corners')
        try:
            latNode = pxFid.get_node('/', latNodeName)
            lonNode = pxFid.get_node('/', lonNodeName)
            # Note: it is assumed that there are no missing values.
            lat = latNode[:].transpose((1,2,0))
            lon = lonNode[:].transpose((1,2,0))            
        finally:
            pxFid.close()
        ind = numpy.indices(lat.shape[0:2]).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
    
    def get_geo_centers(self):
        lat = self.get('Latitude')
        lon = self.get('Longitude')
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape, dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

# Author: Eliot Kim
# Start Date: 8-5-2020
class NETCDFqa4ecvomihchol2_File(NETCDFFile):

    __mainPath = "/PRODUCT/"
    __dataPath = "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
    __geoPath = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/"
    __inputPath = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"

    _nameExpMap = {"amf_clear":__dataPath + "amf_clear",
                   "averaging_kernel_clear":__dataPath + "averaging_kernel_clear",
                   "cloud_radiance_fraction_hcho":__dataPath + "cloud_radiance_fraction_hcho",
                   "intensity_offset_a":__dataPath + "intensity_offset_a",
                   "intensity_offset_a_uncertainty":__dataPath + "intensity_offset_a_uncertainty",
                   "intensity_offset_b":__dataPath + "intensity_offset_b",
                   "intensity_offset_b_uncertainty":__dataPath + "intensity_offset_b_uncertainty",
                   "irradiance_calibration_offset":__dataPath + "irradiance_calibration_offset",
                   "irradiance_calibration_stretch":__dataPath + "irradiance_calibration_stretch",
                   "irradiance_calibration_wavelength":__dataPath + "irradiance_calibration_wavelength",
                   "number_of_spectral_points_in_retrieval":__dataPath + "number_of_spectral_points_in_retrieval",
                   "polynomial_coefficients":__dataPath + "polynomial_coefficients",
                   "polynomial_coefficients_uncertainty":__dataPath + "polynomial_coefficients_uncertainty",
                   "processing_quality_flags":__dataPath + "processing_quality_flags",
                   "radiance_calibration_offset":__dataPath + "radiance_calibration_offset",
                   "radiance_calibration_offset_uncertainty":__dataPath + "radiance_calibration_offset_uncertainty",
                   "radiance_calibration_stretch":__dataPath + "radiance_calibration_stretch",
                   "radiance_calibration_stretch_uncertainty":__dataPath + "radiance_calibration_stretch_uncertainty",
                   "radiance_calibration_wavelength":__dataPath + "radiance_calibration_wavelength",
                   "ring_coefficient":__dataPath + "ring_coefficient",
                   "ring_coefficient_uncertainty":__dataPath + "ring_coefficient_uncertainty",
                   "rms_fit":__dataPath + "rms_fit",
                   "scd_bro":__dataPath + "scd_bro",
                   "scd_bro_uncertainty":__dataPath + "scd_bro_uncertainty",
                   "scd_no2":__dataPath + "scd_no2",
                   "scd_no2_uncertainty":__dataPath + "scd_no2_uncertainty",
                   "scd_o3_223":__dataPath + "scd_o3_223",
                   "scd_o3_223_uncertainty":__dataPath + "scd_o3_223_uncertainty",
                   "scd_o3_243":__dataPath + "scd_o3_243",
                   "scd_o3_243_uncertainty":__dataPath + "scd_o3_243_uncertainty",
                   "scd_o3_lambda":__dataPath + "scd_o3_lambda",
                   "scd_o3_lambda_uncertainty":__dataPath + "scd_o3_lambda_uncertainty",
                   "scd_o3_squared":__dataPath + "scd_o3_squared",
                   "scd_o3_squared_uncertainty":__dataPath + "scd_o3_squared_uncertainty",
                   "scd_o4":__dataPath + "scd_o4",
                   "scd_o4_uncertainty":__dataPath + "scd_o4_uncertainty",
                   "scd_resol":__dataPath + "scd_resol",
                   "scd_resol_uncertainty":__dataPath + "scd_resol_uncertainty",
                   "scd_hcho":__dataPath + "scd_hcho",
                   "scd_hcho_uncertainty_random":__dataPath + "scd_hcho_uncertainty_random",
                   "vcd_hcho_correction_uncertainty":__dataPath + "vcd_hcho_correction_uncertainty",
                   "amf_albedo_uncertainty":__dataPath + "amf_albedo_uncertainty",
                   "amf_cloud_fraction_uncertainty":__dataPath + "amf_cloud_fraction_uncertainty",
                   "amf_cloud_pressure_uncertainty":__dataPath + "amf_cloud_pressure_uncertainty",
                   "amf_a_priori_profile_uncertainty":__dataPath + "amf_a_priori_profile_uncertainty",
                   "profile_height":__dataPath + "profile_height",
                   "scd_hcho_corrected":__dataPath + "scd_hcho_corrected",
                   "scd_hcho_correction":__dataPath + "scd_hcho_correction",
                   "scd_hcho_uncertainty_systematic":__dataPath + "scd_hcho_uncertainty_systematic",
                   "vcd_hcho_correction":__dataPath + "vcd_hcho_correction",
                   "tm5_vcd_hcho_background":__dataPath + "tm5_vcd_hcho_background",
                   "amf_uncertainty":__dataPath + "amf_uncertainty",
                   "tropospheric_hcho_vertical_column_uncertainty_systematic_scdes":__dataPath + "tropospheric_hcho_vertical_column_uncertainty_systematic_scdes",
                   "tropospheric_hcho_vertical_column_uncertainty_systematic_amfes":__dataPath + "tropospheric_hcho_vertical_column_uncertainty_systematic_amfes",
                   "tropospheric_hcho_vertical_column_uncertainty_systematic_vcd0es":__dataPath + "tropospheric_hcho_vertical_column_uncertainty_systematic_vcd0es",
                   "latitude_bounds":__geoPath + "latitude_bounds",
                   "longitude_bounds":__geoPath + "longitude_bounds",
                   "pixel_type":__geoPath + "pixel_type",
                   "relative_azimuth_angle":__geoPath + "relative_azimuth_angle",
                   "relative_azimuth_angle_sat":__geoPath + "relative_azimuth_angle_sat",
                   "satellite_altitude":__geoPath + "satellite_altitude",
                   "satellite_latitude":__geoPath + "satellite_latitude",
                   "satellite_longitude":__geoPath + "satellite_longitude",
                   "solar_zenith_angle":__geoPath + "solar_zenith_angle",
                   "solar_zenith_angle_sat":__geoPath + "solar_zenith_angle_sat",
                   "viewing_zenith_angle":__geoPath + "viewing_zenith_angle",
                   "viewing_zenith_angle_sat":__geoPath + "viewing_zenith_angle_sat",
                   "cloud_fraction":__inputPath + "cloud_fraction",
                   "cloud_fraction_uncertainty":__inputPath + "cloud_fraction_uncertainty",
                   "cloud_pressure":__inputPath + "cloud_pressure",
                   "cloud_pressure_uncertainty":__inputPath + "cloud_pressure_uncertainty",
                   "cloud_albedo":__inputPath + "cloud_albedo",
                   "cloud_albedo_uncertainty":__inputPath + "cloud_albedo_uncertainty",
                   "snow_ice_flag":__inputPath + "snow_ice_flag",
                   "surface_albedo":__inputPath + "surface_albedo",
                   "surface_albedo_hcho":__inputPath + "surface_albedo_hcho",
                   "surface_altitude":__inputPath + "surface_altitude",
                   "surface_altitude_uncertainty":__inputPath + "surface_altitude_uncertainty",
                   "surface_classification":__inputPath + "surface_classification",
                   "surface_pressure":__inputPath + "surface_pressure",
                   "hcho_profile_apriori":__inputPath + "hcho_profile_apriori",
                   "omi_xtrack_flags":__inputPath + "omi_xtrack_flags",
                   "amf_trop":__mainPath + "amf_trop",
                   "averaging_kernel":__mainPath + "averaging_kernel",
                   "tm5_pressure_level_a":__mainPath + "tm5_pressure_level_a",
                   "tm5_pressure_level_b":__mainPath + "tm5_pressure_level_b",
                   "tm5_tropopause_layer_index":__mainPath + "tm5_tropopause_layer_index",
                   "tm5_surface_pressure":__mainPath + "tm5_surface_pressure",
                   "processing_error_flag":__mainPath + "processing_error_flag",
                   "time_utc":__mainPath + "time_utc",
                   "latitude":__mainPath + "latitude",
                   "longitude":__mainPath + "longitude",
                   "delta_time":__mainPath + "delta_time",
                   "tropospheric_hcho_vertical_column":__mainPath + "tropospheric_hcho_vertical_column",
                   "tropospheric_hcho_vertical_column_uncertainty_random":__mainPath + "tropospheric_hcho_vertical_column_uncertainty_random",
                   "tropospheric_hcho_vertical_column_uncertainty_systematic":__mainPath + "tropospheric_hcho_vertical_column_uncertainty_systematic"}

    # Some of the indexing may be incorrect --> needs testing
    _indexMap = {"default":lambda var, ind: var[..., ind[0], ind[1]],
                 "averaging_kernel":lambda var, ind: var[:, ind[0], ind[1], :],
                 "tm5_pressure_level_a":lambda var, ind: var[:],
                 "tm5_pressure_level_b":lambda var, ind: var[:],
                 "time_utc":lambda var, ind: var[:, ind[0]],
                 "delta_time":lambda var, ind: var[ind[0]],
                 "irradiance_calibration_offset":lambda var, ind: var[ind[0]],
                 "irradiance_calibration_stretch":lambda var, ind: var[ind[0]],
                 "irradiance_calibration_wavelength":lambda var, ind: var[ind[0]],
                 "radiance_calibration_wavelength":lambda var, ind: var[ind[0]],
                 "satellite_altitude":lambda var, ind: var[ind[0]],
                 "satellite_latitude":lambda var, ind: var[ind[0]],
                 "satellite_longitude":lambda var, ind: var[ind[0]]}

    def get_geo_corners(self):

        lat = self.get("latitude_bounds")
        lon = self.get("longitude_bounds")

        lat = numpy.squeeze(lat)
        lon = numpy.squeeze(lon)

        ind = numpy.indices(lat.shape[0:2]).transpose((1, 2, 0))

        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

    def get_geo_centers(self):
        lat = self.get("latitude")
        lon = self.get("longitude")
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape,dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

# Author: Eliot Kim
# Start Date: 9/21/2020
class NETCDFtropomino2l2_File(NETCDFFile):

    __mainPath = "/PRODUCT/"
    __geoPath = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/"
    __inputPath = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
    __dataPath = "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"

    _nameExpMap = {"latitude":__mainPath + "latitude",
                    "longitude":__mainPath + "longitude",
                    "delta_time":__mainPath + "delta_time",
                    "time_utc":__mainPath + "time_utc",
                    "qa_value":__mainPath + "qa_value",
                    "nitrogendioxide_tropospheric_column":__mainPath + "nitrogendioxide_tropospheric_column",
                    "nitrogendioxide_tropospheric_column_precision":__mainPath + "nitrogendioxide_tropospheric_column_precision",
                    "nitrogendioxide_tropospheric_column_precision_kernel":__mainPath + "nitrogendioxide_tropospheric_column_precision_kernel",
                    "averaging_kernel":__mainPath + "averaging_kernel",
                    "air_mass_factor_troposphere":__mainPath + "air_mass_factor_troposphere",
                    "air_mass_factor_total":__mainPath + "air_mass_factor_total",
                    "tm5_tropopause_layer_index":__mainPath + "tm5_tropopause_layer_index",
                    "tm5_constant_a":__mainPath + "tm5_constant_a",
                    "tm5_constant_b":__mainPath + "tm5_constant_b",
                    "satellite_latitude":__geoPath + "satellite_latitude",
                    "satellite_longitude":__geoPath + "satellite_longitude",
                    "satellite_altitude":__geoPath + "satellite_altitude",
                    "satellite_orbit_phase":__geoPath + "satellite_orbit_phase",
                    "solar_zenith_angle":__geoPath + "solar_zenith_angle",
                    "solar_azimuth_angle":__geoPath + "solar_azimuth_angle",
                    "viewing_zenith_angle":__geoPath + "viewing_zenith_angle",
                    "latitude_bounds":__geoPath + "latitude_bounds",
                    "longitude_bounds":__geoPath + "longitude_bounds",
                    "geolocation_flags":__geoPath + "geolocation_flags",
                    "surface_altitude":__inputPath + "surface_altitude",
                    "surface_altitude_precision":__inputPath + "surface_altitude_precision",
                    "surface_classification":__inputPath + "surface_classification",
                    "instrument_configuration_identifier":__inputPath + "instrument_configuration_identifier",
                    "instrument_configuration_version":__inputPath + "instrument_configuration_version",
                    "scaled_small_pixel_variance":__inputPath + "scaled_small_pixel_variance",
                    "eastward_wind":__inputPath + "eastward_wind",
                    "northward_wind":__inputPath + "northward_wind",
                    "surface_pressure":__inputPath + "surface_pressure",
                    "surface_albedo_nitrogendioxide_window":__inputPath + "surface_albedo_nitrogendioxide_window",
                    "surface_albedo":__inputPath + "surface_albedo",
                    "cloud_pressure_crb":__inputPath + "cloud_pressure_crb",
                    "cloud_fraction_crb":__inputPath + "cloud_fraction_crb",
                    "scene_albedo":__inputPath + "scene_albedo",
                    "apparent_scene_pressure":__inputPath + "apparent_scene_pressure",
                    "snow_ice_flag":__inputPath + "snow_ice_flag",
                    "aerosol_index_354_388":__inputPath + "aerosol_index_354_388",
                    "processing_quality_flags":__dataPath + "processing_quality_flags",
                    "number_of_spectral_points_in_retrieval":__dataPath + "number_of_spectral_points_in_retrieval",
                    "number_of_iterations":__dataPath + "number_of_iterations",
                    "wavelength_calibration_offset":__dataPath + "wavelength_calibration_offset",
                    "wavelength_calibration_offset_precision":__dataPath + "wavelength_calibration_offset_precision",
                    "wavelength_calibration_stretch":__dataPath + "wavelength_calibration_stretch",
                    "wavelength_calibration_stretch_precision":__dataPath + "wavelength_calibration_stretch_precision",
                    "wavelength_calibration_chi_square":__dataPath + "wavelength_calibration_chi_square",
                    "wavelength_calibration_irradiance_offset":__dataPath + "wavelength_calibration_irradiance_offset",
                    "wavelength_calibration_irradince_offset_precision":__dataPath + "wavelength_calibration_irradince_offset_precision",
                    "wavelength_calibration_irradiance_chi_square":__dataPath + "wavelength_calibration_irradiance_chi_square",
                    "nitrogendioxide_stratospheric_column":__dataPath + "nitrogendioxide_stratospheric_column",
                    "nitrogendioxide_stratospheric_column_precision":__dataPath + "nitrogendioxide_stratospheric_column_precision",
                    "nitrogendioxide_total_column":__dataPath + "nitrogendioxide_total_column",
                    "nitrogendioxide_total_column_precision":__dataPath + "nitrogendioxide_total_column_precision",
                    "nitrogendioxide_total_column_precision_kernel":__dataPath + "nitrogendioxide_total_column_precision_kernel",
                    "nitrogendioxide_summed_total_column":__dataPath + "nitrogendioxide_summed_total_column",
                    "nitrogendioxide_summed_total_column_precision":__dataPath + "nitrogendioxide_summed_total_column_precision",
                    "nitrogendioxide_slant_column_density":__dataPath + "nitrogendioxide_slant_column_density",
                    "nitrogendioxide_slant_column_density_precision":__dataPath + "nitrogendioxide_slant_column_density_precision",
                    "nitrogendioxide_slant_column_density_stripe_amplitude":__dataPath + "nitrogendioxide_slant_column_density_stripe_amplitude",
                    "ozone_slant_column_density":__dataPath + "ozone_slant_column_density",
                    "ozone_slant_column_density_precision":__dataPath + "ozone_slant_column_density_precision",
                    "oxygen_oxygen_dimer_slant_column_density":__dataPath + "oxygen_oxygen_dimer_slant_column_density",
                    "oxygen_oxygen_dimer_slant_column_density_precision":__dataPath + "oxygen_oxygen_dimer_slant_column_density_precision",
                    "water_slant_column_density":__dataPath + "water_slant_column_density",
                    "water_slant_column_density_precision":__dataPath + "water_slant_column_density_precision",
                    "water_liquid_slant_column_density":__dataPath + "water_liquid_slant_column_density",
                    "water_liquid_slant_column_density_precision":__dataPath + "water_liquid_slant_column_density_precision",
                    "ring_coefficient":__dataPath + "ring_coefficient",
                    "ring_coefficient_precision":__dataPath + "ring_coefficient_precision",
                    "polynomial_coefficients":__dataPath + "polynomial_coefficients",
                    "polynomial_coefficients_precision":__dataPath + "polynomial_coefficients_precision",
                    "intensity_offset_coefficients":__dataPath + "intensity_offset_coefficients",
                    "intensity_offset_coefficients_precision":__dataPath + "intensity_offset_coefficients_precision",
                    "cloud_fraction_crb_nitrogendioxide_window":__dataPath + "cloud_fraction_crb_nitrogendioxide_window",
                    "cloud_radiance_fraction_nitrogendioxide_window":__dataPath + "cloud_radiance_fraction_nitrogendioxide_window",
                    "chi_square":__dataPath + "chi_square",
                    "root_mean_square_error_of_fit":__dataPath + "root_mean_square_error_of_fit",
                    "degrees_of_freedom":__dataPath + "degrees_of_freedom",
                    "air_mass_factor_stratosphere":__dataPath + "air_mass_factor_stratosphere",
                    "air_mass_factor_cloudy":__dataPath + "air_mass_factor_cloudy",
                    "air_mass_factor_clear":__dataPath + "air_mass_factor_clear",
                    "nitrogendioxide_ghost_column":__dataPath + "nitrogendioxide_ghost_column"}

    _indexMap = {"default":lambda var, ind: var[..., ind[0], ind[1]],
                 "delta_time":lambda var, ind: var[:, ind[0]],
                 "time_utc":lambda var, ind: var[:, ind[0]],
                 "averaging_kernel":lambda var, ind: var[:, ind[0], ind[1], :],
                 "tm5_constant_a":lambda var, ind: var[:],
                 "tm5_constant_b":lambda var, ind: var[:],
                 "satellite_latitude":lambda var, ind: var[:, ind[0]],
                 "satellite_longitude":lambda var, ind: var[:, ind[0]],
                 "satellite_altitude":lambda var, ind: var[:, ind[0]],
                 "satellite_orbit_phase":lambda var, ind: var[:, ind[0]],
                #  "latitude_bounds":lambda var, ind: 
                #  "longitude_bounds":lambda var, ind:
                 "instrument_configuration_identifier":lambda var, ind: var[:, ind[0]],
                 "instrument_configuration_version":lambda var, ind: var[:, ind[0]],
                 "wavelength_calibration_irradiance_offset":lambda var, ind: var[:, ind[1]],
                 "wavelength_calibration_irradiance_offset_precision":lambda var, ind: var[:, ind[1]],
                 "wavelength_calibration_irradiance_chi_square":lambda var, ind: var[:, ind[1]],
                 "nitrogendioxide_slant_column_density_stripe_amplitude":lambda var, ind: var[:, ind[1]],
                 "polynomial_coefficients":lambda var, ind: var[:, ind[0], ind[1], :],
                 "polynomial_coefficients_precision":lambda var, ind: var[:, ind[0], ind[1], :],
                 "intensity_offset_coefficients":lambda var, ind: var[:, ind[0], ind[1], :],
                 "intensity_offset_coefficients_precision":lambda var, ind: var[:, ind[0], ind[1], :]}

    def get_geo_corners(self):

        lat = self.get("latitude_bounds")
        lon = self.get("longitude_bounds")

        lat = numpy.squeeze(lat)
        lon = numpy.squeeze(lon)

        ind = numpy.indices(lat.shape[0:2]).transpose((1, 2, 0))

        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

    def get_geo_centers(self):
        lat = self.get("latitude")
        lon = self.get("longitude")
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape,dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

# Author: Eliot Kim
# Start Date: 10/18/2020
class NETCDFtropomiso2l2_File(NETCDFFile):

    __mainPath = "/PRODUCT/"
    __geoPath = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/"
    __inputPath = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
    __dataPath = "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"

    _nameExpMap = {'scanline': __mainPath,
                    'ground_pixel': __mainPath,
                    'time': __mainPath,
                    'corner': __mainPath,
                    'latitude': __mainPath,
                    'longitude': __mainPath,
                    'delta_time': __mainPath,
                    'time_utc': __mainPath,
                    'qa_value': __mainPath,
                    'sulfurdioxide_total_vertical_column': __mainPath,
                    'sulfurdioxide_total_vertical_column_precision': __mainPath,
                    'layer': __mainPath,
                    'satellite_latitude': __geoPath,
                    'satellite_longitude': __geoPath,
                    'satellite_altitude': __geoPath,
                    'satellite_orbit_phase': __geoPath,
                    'solar_zenith_angle': __geoPath,
                    'solar_azimuth_angle': __geoPath,
                    'viewing_zenith_angle': __geoPath,
                    'viewing_azimuth_angle': __geoPath,
                    'latitude_bounds': __geoPath,
                    'longitude_bounds': __geoPath,
                    'geolocation_flags': __geoPath,
                    'snow_ice_flag_nise': __inputPath,
                    'snow_ice_flag': __inputPath,
                    'cloud_fraction_crb': __inputPath,
                    'cloud_fraction_crb_precision': __inputPath,
                    'cloud_pressure_crb': __inputPath,
                    'cloud_pressure_crb_precision': __inputPath,
                    'cloud_height_crb': __inputPath,
                    'cloud_height_crb_precision': __inputPath,
                    'cloud_albedo_crb': __inputPath,
                    'cloud_albedo_crb_precision': __inputPath,
                    'ozone_total_vertical_column': __inputPath,
                    'ozone_total_vertical_column_precision': __inputPath,
                    'surface_altitude': __inputPath,
                    'surface_altitude_precision': __inputPath,
                    'surface_classification': __inputPath,
                    'instrument_configuration_identifier': __inputPath,
                    'instrument_configuration_version': __inputPath,
                    'scaled_small_pixel_variance': __inputPath,
                    'surface_pressure': __inputPath,
                    'tm5_constant_a': __inputPath, # Just 1 column of TM5 Pressure Level values instead of 2 as in QA4ECV and TROPOMI NO2.
                    'tm5_constant_b': __inputPath,
                    'aerosol_index_340_380': __inputPath,
                    'surface_albedo_328nm': __inputPath,
                    'surface_albedo_376nm': __inputPath,
                    'sulfurdioxide_total_vertical_column_trueness': __dataPath,
                    'fitted_slant_columns_win1': __dataPath,
                    'fitted_slant_columns_win1_precision': __dataPath,
                    'fitted_slant_columns_win2': __dataPath,
                    'fitted_slant_columns_win2_precision': __dataPath,
                    'fitted_slant_columns_win3': __dataPath,
                    'fitted_slant_columns_win3_precision': __dataPath,
                    'selected_fitting_window_flag': __dataPath,
                    'sulfurdioxide_slant_column_corrected': __dataPath,
                    'sulfurdioxide_slant_column_corrected_trueness': __dataPath,
                    'sulfurdioxide_slant_column_corrected_win1': __dataPath,
                    'sulfurdioxide_slant_column_corrected_win2': __dataPath,
                    'sulfurdioxide_slant_column_corrected_win3': __dataPath,
                    'fitted_root_mean_square_win1': __dataPath,
                    'fitted_root_mean_square_win2': __dataPath,
                    'fitted_root_mean_square_win3': __dataPath,
                    'averaging_kernel': __dataPath,
                    'sulfurdioxide_profile_apriori': __dataPath,
                    'fitted_radiance_shift': __dataPath,
                    'fitted_radiance_squeeze': __dataPath,
                    'fitted_radiance_shift_win1': __dataPath,
                    'fitted_radiance_squeeze_win1': __dataPath,
                    'number_of_spectral_points_in_retrieval_win1': __dataPath,
                    'fitted_radiance_shift_win2': __dataPath,
                    'fitted_radiance_squeeze_win2': __dataPath,
                    'number_of_spectral_points_in_retrieval_win2': __dataPath,
                    'fitted_radiance_shift_win3': __dataPath,
                    'fitted_radiance_squeeze_win3': __dataPath,
                    'number_of_spectral_points_in_retrieval_win3': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_polluted': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_polluted_precision': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_polluted_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_polluted_kernel_trueness': __dataPath,
                    'sulfurdioxide_clear_air_mass_factor_polluted': __dataPath,
                    'sulfurdioxide_cloudy_air_mass_factor_polluted': __dataPath,
                    'sulfurdioxide_total_vertical_column_1km': __dataPath,
                    'sulfurdioxide_total_vertical_column_1km_precision': __dataPath,
                    'sulfurdioxide_total_vertical_column_1km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_1km': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_1km_precision': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_1km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_1km_kernel_trueness': __dataPath,
                    'sulfurdioxide_clear_air_mass_factor_1km': __dataPath,
                    'sulfurdioxide_cloudy_air_mass_factor_1km': __dataPath,
                    'sulfurdioxide_averaging_kernel_scaling_box_1km': __dataPath,
                    'sulfurdioxide_total_vertical_column_7km': __dataPath,
                    'sulfurdioxide_total_vertical_column_7km_precision': __dataPath,
                    'sulfurdioxide_total_vertical_column_7km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_7km': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_7km_precision': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_7km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_7km_kernel_trueness': __dataPath,
                    'sulfurdioxide_clear_air_mass_factor_7km': __dataPath,
                    'sulfurdioxide_cloudy_air_mass_factor_7km': __dataPath,
                    'sulfurdioxide_averaging_kernel_scaling_box_7km': __dataPath,
                    'sulfurdioxide_total_vertical_column_15km': __dataPath,
                    'sulfurdioxide_total_vertical_column_15km_precision': __dataPath,
                    'sulfurdioxide_total_vertical_column_15km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_15km': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_15km_precision': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_15km_trueness': __dataPath,
                    'sulfurdioxide_total_air_mass_factor_15km_kernel_trueness': __dataPath,
                    'sulfurdioxide_clear_air_mass_factor_15km': __dataPath,
                    'sulfurdioxide_cloudy_air_mass_factor_15km': __dataPath,
                    'sulfurdioxide_averaging_kernel_scaling_box_15km': __dataPath,
                    'sulfurdioxide_slant_column_correction_flag': __dataPath,
                    'sulfurdioxide_detection_flag': __dataPath,
                    'number_of_iterations_in_retrieval': __dataPath,
                    'number_of_iterations_in_retrieval_win1': __dataPath,
                    'number_of_iterations_in_retrieval_win2': __dataPath,
                    'number_of_iterations_in_retrieval_win3': __dataPath,
                    'fitted_root_mean_square': __dataPath,
                    'cloud_fraction_intensity_weighted': __dataPath,
                    'cloud_fraction_intensity_weighted_precision': __dataPath,
                    'processing_quality_flags': __dataPath,
                    'number_of_spectral_points_in_retrieval': __dataPath,
                    'number_of_slant_columns_win1': __dataPath,
                    'number_of_slant_columns_win2': __dataPath,
                    'number_of_slant_columns_win3': __dataPath}

    _indexMap = {"default":lambda var, ind: var[..., ind[0], ind[1]],
                 "time_utc":lambda var, ind: var[:, ind[0]],
                 "satellite_latitude":lambda var, ind: var[:, ind[0]],
                 "satellite_longitude":lambda var, ind: var[:, ind[0]],
                 "satellite_altitude":lambda var, ind: var[:, ind[0]],
                 "satellite_orbit_phase":lambda var, ind: var[:, ind[0]],
                 "instrument_configuration_identifier":lambda var, ind: var[:, ind[0]],
                 "instrument_configuration_version":lambda var, ind: var[:, ind[0]],
                 "tm5_constant_a":lambda var, ind: var[:],
                 "tm5_constant_b":lambda var, ind: var[:],
                 "fitted_slant_columns_win1":lambda var, ind: var[:, ind[0], ind[1], :],
                 "fitted_slant_columns_win1_precision":lambda var, ind: var[:, ind[0], ind[1], :],
                 "fitted_slant_columns_win2":lambda var, ind: var[:, ind[0], ind[1], :],
                 "fitted_slant_columns_win2_precision":lambda var, ind: var[:, ind[0], ind[1], :],
                 "fitted_slant_columns_win3":lambda var, ind: var[:, ind[0], ind[1], :],
                 "fitted_slant_columns_win3_precision":lambda var, ind: var[:, ind[0], ind[1], :],
                 "averaging_kernel":lambda var, ind: var[:, ind[0], ind[1], :],
                 "sulfurdioxide_profile_apriori":lambda var, ind: var[:, ind[0], ind[1], :]}

    def get_geo_corners(self):

        lat = self.get("latitude_bounds")
        lon = self.get("longitude_bounds")

        lat = numpy.squeeze(lat)
        lon = numpy.squeeze(lon)

        ind = numpy.indices(lat.shape[0:2]).transpose((1, 2, 0))

        protoDtype = [('lat', lat.dtype, 4), ('lon', lon.dtype, 4), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape[0:2], dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct

    def get_geo_centers(self):
        lat = self.get("latitude")
        lon = self.get("longitude")
        ind = numpy.indices(lat.shape).transpose((1,2,0))
        protoDtype = [('lat', lat.dtype), ('lon', lon.dtype), ('ind', ind.dtype, 2)]
        struct = numpy.zeros(lat.shape,dtype=protoDtype)
        (struct['lat'], struct['lon'], struct['ind']) = (lat, lon, ind)
        return struct
                                     
