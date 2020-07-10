'''
Framework for remapping data.

All functions are named after the convention 
scheme_map_geo.  The function ValidMaps() automatically
maintains a list of implemented mapping schemes.

The interface for schemes is set up to allow
for maximum efficiency of the computations within
the scheme.  They therefore take only a parser and
a griddef (both instances, in that order)
as arguments, and return a dictionary.

The returned dictionary must be formatted as follows:
    -one key "parser" which corresponds to a pointer
    to the parser used to generate the map.
    -one key for each of the gridboxes, which is a tuple
    (row,col) for that gridbox.  These keys correspond
    to lists.  Each list contains tuples, each of which 
    must contain:
        -a tuple of the SWATH indices (this should
        be able to be fed into the get function of 
        the parser as is)
        - the data-independent weight (pass None if
        no weight is computed from this function)
'''
import sys
from itertools import izip
import datetime
import pdb

import map_helpers
import utils
import grid_geo

from shapely.prepared import prep
import shapely.geometry as geom
import numpy

def ValidMaps():
    '''Return a list of valid map names'''
    currentModule = sys.modules[__name__]
    names = dir(currentModule)
    return [el[:-8] for el in names if el.endswith("_map_geo")]

def global_intersect_map_geo(parser, griddef, verbose=True):
    '''
    For each pixel, find all gridcells that it intersects

    This function does not compute or save the fractional 
    overlap of individual pixels.  

    This function is designed to handle grids that span 
    the entire globe, with the cyclic point (the point 
    where longitude "wraps") ocurring at index [*,0]

    The function assumes that the projection performs the wrapping.  
    IE any pixels east of the far east edge of the domain will be
    on the west side of the projection.  The function will NOT
    automatically wrap pixels that fall off the east/west edges
    of the projection.

    Pixels with NaN for any vertex are rejected

    Assumptions:
        - Straight lines in projected space adequately 
        approximate the edges of pixels/gridcells.
        - polar discontinuities aren't of concern
        - we ARE dealing with a global projection that 
        cyclizes at index (*,0)
        - grid is rectilinear
        - pixels are convex polygons
    '''
    if verbose:
        print('Mapping '+parser.name+'\nat '+str(datetime.datetime.now()))  
    outer_indices = griddef.indLims()
    # create the dictionary we'll use as a map
    map = map_helpers.init_output_map(outer_indices)
    map['parser'] = parser
    # we're going to hold onto both the prepared and unprepared versions
    # of the polys, so we can access the fully method set in the unprep
    # polys, but still do fast comparisons
    gridPolys = map_helpers.rect_grid_polys(outer_indices)
    prepPolys = map_helpers.rect_grid_polys(outer_indices)
    if verbose: print('prepping polys in grid')
    # prepare polygons, they're going to get compared a lot
    for polykey in prepPolys.keys():
        prepPolys[polykey] = prep(prepPolys[polykey])
    if verbose: print('done prepping polys in grid')
    cornersStruct = parser.get_geo_corners()
    (row, col) = griddef.geoToGridded(cornersStruct['lat'], \
                                      cornersStruct['lon']) 
    ind = cornersStruct['ind']
    # reshape the matrixes to make looping workable
    row = row.reshape(-1,4)
    col = col.reshape(-1,4)
    ind = ind.reshape(row.shape[0],-1)
    if verbose: print('Intersecting pixels')
    # create the appropriate pixel(s) depending on whether
    # the pixels span the cyclic point
    minCol = griddef.indLims()[2]
    maxCol = griddef.indLims()[3] + 1
    midCol = (minCol+maxCol)/2.0
    for (pxrow, pxcol, pxind) in izip(row, col, ind):
        if (numpy.any(numpy.isnan(pxrow)) or 
            numpy.any(numpy.isnan(pxcol))):
            continue # skip incomplete pixels
        pointsTup = zip(pxrow, pxcol)
        prelimPoly = geom.MultiPoint(pointsTup).convex_hull
        (bbBot, bbLeft, bbTop, bbRight) = prelimPoly.bounds
        if bbLeft < midCol and bbRight > midCol:
            pointsLeft = [ (r,c) for (r,c) in pointsTup if c < midCol]
            pointsRight = [ (r,c) for (r,c) in pointsTup if c >= midCol]
            pointsLeft += [ (bbBot, minCol), (bbTop, minCol) ]
            pointsRight += [ (bbBot, maxCol), (bbTop, maxCol) ]
            polyLeft = geom.MultiPoint(pointsLeft).convex_hull
            polyRight = geom.MultiPoint(pointsRight).convex_hull
            splitArea = polyLeft.area + polyRight.area
            spanArea = prelimPoly.area
            if splitArea < spanArea:
                pixPolys = [polyLeft, polyRight]
            else:
                pixPolys = [prelimPoly]
        else:
            pixPolys = [prelimPoly]
        # try intersecting the poly(s) with all the grid polygons
        for poly in pixPolys:
            for key in map_helpers.get_possible_cells(outer_indices, poly):
                if prepPolys[key].intersects(poly) and not gridPolys[key].touches(poly):
                    map[key].append((tuple(pxind), None))
    if verbose: print('Done intersecting.')
    return map

    
def regional_intersect_map_geo(parser, griddef, verbose=True):
    '''
    For each pixel, find all gridcells that it intersects

    This function does not compute or save the fractional
    overlap of individual pixels.  It simply stores the
    pixel indices themselves.
    
    This function is currently not configured to operate 
    on a global scale, or near discontinuities. 
    The current kludge to handle discontinuities is 
    relies on the assumption that any pixel of interest 
    will have at least one corner within the bounds of 
    the grid

    Pixels with NaN for any vertex are rejected
    
    Several assumptions are made:
        - Straight lines in projected space adequately 
        approximate the edges of pixels/gridcells.
        - polar discontinuities aren't of concern
        - we AREN'T dealing with a global projection
        - grid is rectilinear
        - pixels are convex polygons
    '''
    
    if verbose:
        print('Mapping '+parser.name+'\nat '+str(datetime.datetime.now()))
    outer_indices = griddef.indLims()
    map = map_helpers.init_output_map(outer_indices)
    map['parser'] = parser
    bounds = prep(map_helpers.rect_bound_poly(outer_indices))

    # we're going to hold onto both the prepared and unprepared versions
    # of the polys, so we can access the fully method set in the unprep
    # polys, but still do fast comparisons
    gridPolys = map_helpers.rect_grid_polys(outer_indices)
    prepPolys = map_helpers.rect_grid_polys(outer_indices)
    if verbose: print('prepping polys in grid')
    # prepare polygons, they're going to get compared a lot
    for polykey in prepPolys.keys():
        prepPolys[polykey] = prep(prepPolys[polykey])  
    if verbose: print('done prepping polys in grid')
    cornersStruct = parser.get_geo_corners()
    (row, col) = griddef.geoToGridded(cornersStruct['lat'], \
                                      cornersStruct['lon'])
    ind = cornersStruct['ind']
    # reshape the matrixes to make looping workable
    row = row.reshape(-1,4)
    col = col.reshape(-1,4)
    ind = ind.reshape(row.shape[0],-1)
    if verbose:
        griddedPix = 0 
        print('Intersecting pixels')
        sys.stdout.write("Approximately 0 pixels gridded. ")
        sys.stdout.flush()
        for (pxrow, pxcol, pxind) in izip(row, col, ind):
            if numpy.any(numpy.isnan(pxrow)) or numpy.any(numpy.isnan(pxcol)):
                continue  # if we have only a partial pixel, skip
            elif not any([bounds.contains(geom.asPoint((r,c))) \
                       for (r,c) in izip(pxrow, pxcol)]):
                continue  # if none of the corners are in bounds, skip
            griddedPix += 1
            sys.stdout.write("\rApproximately {0} pixels gridded. ".\
                             format(griddedPix))
            sys.stdout.flush()
            pixPoly = geom.MultiPoint(zip(pxrow, pxcol)).convex_hull
            
            for key in map_helpers.get_possible_cells(outer_indices, pixPoly):
                if prepPolys[key].intersects(pixPoly) and not \
                                  gridPolys[key].touches(pixPoly) :
                    map[key].append((tuple(pxind), None))
        print('Done intersecting.')
    else:
        for (pxrow, pxcol, pxind) in izip(row, col, ind):
            if numpy.any(numpy.isnan(pxrow)) or numpy.any(numpy.isnan(pxcol)):
                continue  # if we have only a partial pixel, skip
            elif not any([bounds.contains(geom.asPoint((r,c))) for (r,c) \
                                        in izip(pxrow, pxcol)]):
                continue  # if none of the corners are in bounds, skip
            pixPoly = geom.MultiPoint(zip(pxrow, pxcol)).convex_hull
            for key in map_helpers.get_possible_cells(outer_indices, pixPoly):
                if prepPolys[key].intersects(pixPoly) and not \
                                  gridPolys[key].touches(pixPoly):
                    map[key].append((tuple(pxind), None))
    return map

def point_in_cell_map_geo(parser, griddef, verbose=True):
    '''
    For each object, find the single cell to which it should be assigned.  This 
    cell is determined as the cell into which the representative lat/lon of the 
    pixel would fit in projected space.
    
    Cells are treated as open on upper right and closed on the lower left.  For
    practical purposes, this means that pixels falling on the boundary between 
    two cells will be assigned to the cell up and/or to their right.  Pixels 
    falling on the lower left boundaries of the griddable area will be assigned
    to a cell and those on the upper right boundaries of the griddable area
    will be discarded.
    
    This function computes no weights.  It simply assigns objects on the basis
    of a representative lat-lon given by the parser's get_geo_centers function.
    
    This can operate on any rectilinear rid.  So long as a lat-lon pair can be
    projected to a unique location, it will assign each pixel to one and only 
    one location.
    
    Several assumptions are made:
        - Straight lines in projected space adequately approximate the edges of
        gridcells.
        - Grid is rectilinear.
        - the lat/lon of the col,row origin is the lower left hand corner of 
        the 0,0 gridbox.
    '''
    if verbose:
        print('Mapping '+parser.name+'\nat '+str(datetime.datetime.now()))
    # Create an empty map to be filled
    mapLims = griddef.indLims()
    map = map_helpers.init_output_map(mapLims)
    map['parser'] = parser
    centersStruct = parser.get_geo_centers()
    # get the data to geolocate the pixels and reshape to make looping feasible
    ind = centersStruct['ind']
    (row, col) = griddef.geoToGridded(centersStruct['lat'], \
                                      centersStruct['lon'])
    row = numpy.floor(row.flatten()) # we floor values to get the cell indices
    col = numpy.floor(col.flatten())
    ind = ind.reshape(row.size, -1)
    # loop over pixels and grid as appropriate
    (minRow, maxRow, minCol, maxCol) = mapLims  # unpack this so we can use it
    if verbose:
        nGriddedPix = 0
        print('Assigning pixels to gridboxes.')
        sys.stdout.write("Approximately 0 pixels gridded. ")
        sys.stdout.flush()
        for (pxrow, pxcol, pxind) in izip(row, col, ind):
            if minRow <= pxrow <= maxRow and minCol <= pxcol <= maxCol:
                map[(pxrow, pxcol)].append((tuple(pxind), None))
                nGriddedPix += 1
                sys.stdout.write("\rApproximately {0} pixels gridded. "\
                                 .format(nGriddedPix))
                sys.stdout.flush()
        print('Done intersecting.')
    else:
        for (pxrow, pxcol, pxind) in izip(row, col, ind):
            if minRow <= pxrow <= maxRow and minCol <= pxcol <= maxCol:
                map[(pxrow, pxcol)].append((tuple(pxind), None))
    return map
       

def OMNO2d_regional_map_geo(parser, griddef, verbose=True):
    '''
    For each pixel, find all gridcells that it intersects

    This function computes the weight for the OMNO2d 
    algorithm and stores the pixel indices.
    
    OMNO2d weighting algorithm can be found in the OMNO2d 
    README file at: 
    <https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMNO2d.003/doc/README.OMNO2.pdf>
        
    This function is currently not configured to operate 
    on a global scale, or near discontinuities. 
    The current kludge to handle discontinuities is 
    relies on the assumption that any pixel of interest 
    will have at least one corner within the bounds of 
    the grid

    Pixels with NaN for any vertex are rejected
    
    Several assumptions are made:
        - Straight lines in projected space adequately 
        approximate the edges of pixels/gridcells.
        - polar discontinuities aren't of concern
        - we AREN'T dealing with a global projection
        - grid is rectilinear
        - pixels are convex polygons
    '''
    
    if verbose:
        print('Mapping '+parser.name+'\nat '+str(datetime.datetime.now()))
    outer_indices = griddef.indLims()
    map = map_helpers.init_output_map(outer_indices)
    areaMap = map_helpers.init_output_map(outer_indices) # omno2d
    map['parser'] = parser
    bounds = prep(map_helpers.rect_bound_poly(outer_indices))
    
    # instantiate an equal area grid (for pixel areas)
    equalAreaGrid = grid_geo.cylequalarea_GridDef(
            {"stdPar": 0.,"refLon": 0.,"xOrig": 0.,"yOrig": 0.,"xCell":0.5,
             "yCell":0.5,"nRows":360,"nCols":720,"earthRadius":6371.})

    # we're going to hold onto both the prepared and unprepared versions
    # of the polys, so we can access the fully method set in the unprep
    # polys, but still do fast comparisons
    gridPolys = map_helpers.rect_grid_polys(outer_indices)
    prepPolys = map_helpers.rect_grid_polys(outer_indices)
    if verbose: print('prepping polys in grid')
    # prepare polygons, they're going to get compared a lot
    for polykey in prepPolys.keys():
        prepPolys[polykey] = prep(prepPolys[polykey])  
    if verbose: print('done prepping polys in grid')
    cornersStruct = parser.get_geo_corners()
    (row, col) = griddef.geoToGridded(cornersStruct['lat'], \
                                      cornersStruct['lon'])
    ind = cornersStruct['ind']
    
    # apply an equal area projection to pixel corners (used for calculating 
    # approximate pixel areas in km^2)
    (equalAreaY, equalAreaX) = equalAreaGrid.geoToProjected(
                                                        cornersStruct['lat'], \
                                                        cornersStruct['lon'])
    # reshape the matrixes to make looping workable
    row = row.reshape(-1,4)
    col = col.reshape(-1,4)
    ind = ind.reshape(row.shape[0],-1)
    equalAreaY = equalAreaY.reshape(-1,4)
    equalAreaX = equalAreaX.reshape(-1,4)
    # initialize min and maxPixArea to nan 
    minPixArea = numpy.nan
    maxPixArea = numpy.nan
    
    #temp
    # todo
    #GOME-2
    minPixArea = 3150.
    maxPixArea = 5800.
#   maxPixArea = 15500.
    
    #OMI
    # todo
#    minPixArea = 250.
#    maxPixArea = 2100.
    print minPixArea
    print maxPixArea
    #end temp
    
    if verbose:
        griddedPix = 0 
        print('Intersecting pixels')
        sys.stdout.write("Approximately 0 pixels gridded. ")
        sys.stdout.flush()
        for (pxrow, pxcol, pxind, pxY, pxX) in izip(row, col, ind, 
                                                    equalAreaY, equalAreaX):
            if numpy.any(numpy.isnan(pxrow)) or numpy.any(numpy.isnan(pxcol)):
                continue  # if we have only a partial pixel, skip
            elif not any([bounds.contains(geom.asPoint((r,c))) \
                       for (r,c) in izip(pxrow, pxcol)]):
                continue  # if none of the corners are in bounds, skip
            # find approx. pixel area (in km^2) using an equal area projection
            pixArea = geom.MultiPoint(zip(pxY, pxX)).convex_hull.area
            # skip pixels which are too big or too small
            if pixArea<minPixArea:
                continue
            if pixArea>maxPixArea:
                continue
            griddedPix += 1
            sys.stdout.write("\rApproximately {0} pixels gridded. ".\
                             format(griddedPix))
            sys.stdout.flush()
            pixPoly = geom.MultiPoint(zip(pxrow, pxcol)).convex_hull
            
            for key in map_helpers.get_possible_cells(outer_indices, pixPoly):
                gridArea = gridPolys[key].area 
                if prepPolys[key].intersects(pixPoly) and not \
                                  gridPolys[key].touches(pixPoly) :
                    overlapArea = pixPoly.intersection(gridPolys[key]).area 
                    areaMap[key].append((tuple(pxind),tuple((pixArea,gridArea,overlapArea))))
            
        #Calculate OMNO2d weights
        # loop again because we don't know min/maxPixArea until end of 1st loop
        if griddedPix != 0: # don't run if nothing gridded
            for (key, areaTup) in areaMap.iteritems(): # loop over gridcells
                for (pxind, areas) in areaTup: # loop over pixels
                    (pixArea, gridArea, overlapArea) = areas
                    wAi = 1. - (pixArea - minPixArea)/maxPixArea
                    Qij = overlapArea / gridArea 
                    weight = wAi * Qij
                    map[key].append((tuple(pxind),weight)) 

        print('Done intersecting.')

    else:

        raise NotImplementedError('Non-verbose version of OMNO2d_regional_map has  not been implemented. Please change to verbose=True')
#        griddedPix = 0 # omno2d
#        for (pxrow, pxcol, pxind, pxY, pxX) in \
#                                  izip(row, col, ind, equalAreaY, equalAreaX):
#            if numpy.any(numpy.isnan(pxrow)) or numpy.any(numpy.isnan(pxcol)):
#                continue  # if we have only a partial pixel, skip
#            elif not any([bounds.contains(geom.asPoint((r,c))) for (r,c) \
#                                        in izip(pxrow, pxcol)]):
#                continue  # if none of the corners are in bounds, skip
#            griddedPix += 1 # omno2d
#            pixPoly = geom.MultiPoint(zip(pxrow, pxcol)).convex_hull
#            pixArea = geom.MultiPoint(zip(pxY, pxX)).convex_hull.area # omno2d
#            # find the pixel with the min/max area in this map
#            minPixArea = numpy.nanmin([minPixArea,pixArea]) # omno2d
#            maxPixArea = numpy.nanmax([maxPixArea,pixArea]) # omno2d
#            for key in map_helpers.get_possible_cells(outer_indices, pixPoly):
#                gridArea = gridPolys[key].area # omno2d
#                if prepPolys[key].intersects(pixPoly) and not \
#                                  gridPolys[key].touches(pixPoly):
#                    overlapArea = pixPoly.intersection(gridPolys[key]).area # omno2d
#                    areaMap[key].append((tuple(pxind),tuple((pixArea,gridArea,overlapArea)))) # omno2d
#        #Calculate OMNO2d weights
#        # loop again because we don't know min/maxPixArea until end of 1st loop
#        if griddedPix != 0: # don't run if nothing gridded
#            for (key, areaTup) in areaMap.iteritems(): # loop over gridcells
#                for (pxind, areas) in areaTup: # loop over pixels
#                    (pixArea, gridArea, overlapArea) = areas
#                    wAi = 1. - (pixArea - minPixArea)/maxPixArea
#                    Qij = overlapArea / gridArea 
#                    weight = wAi * Qij
#                    map[key].append((tuple(pxind),weight)) 
                    
    return map


def area_test_map_geo(parser, griddef, verbose=True):
    '''
    Just for testing pixel areas.
    '''
    
    import grid_geo
    # DOMINO
    # pixRange = (250-2100)
    # GOME-2
    #pixRange = ()

    gridParms = {"stdPar": 0.,"refLon": 0.,"xOrig": 0.,"yOrig": 0.,
    "xCell":0.5,"yCell":0.5,"nRows":360,"nCols":720,"earthRadius":6371.}
    gridCEA = grid_geo.cylequalarea_GridDef(gridParms)
    
    cornersStruct = parser.get_geo_corners()
    (lat, lon, ind) = (cornersStruct['lat'], cornersStruct['lon'], \
                        cornersStruct['ind'])
    lat = lat.reshape(-1,4)
    lon = lon.reshape(-1,4)
    ind = ind.reshape(lat.shape[0],-1)
    pixAreaCylProjList = []
    pixCode = []
    with parser as p:
        for (pxlat, pxlon, pxind) in izip(lat, lon, ind):
            if numpy.any(numpy.isnan(pxlat)) or numpy.any(numpy.isnan(pxlon)):
                continue  # if we have only a partial pixel, skip
            if numpy.ptp(pxlon) > 180.:
                continue # if we're going the wrong way around the sphere, skip
            (yCEA, xCEA) = gridCEA.geoToProjected(pxlat,pxlon)
            pixPolyCyl = geom.MultiPoint(zip(yCEA, xCEA)).convex_hull
            pixProjAreaCyl = pixPolyCyl.area 
            pixAreaCylProjList.append(pixProjAreaCyl)
            pixCode.append(p.get_cm('fltrop',indices=pxind)) #fltrop o/w
        
    maxPix = numpy.array(numpy.max(pixAreaCylProjList)).reshape(1,)
    minPix = numpy.array(numpy.min(pixAreaCylProjList)).reshape(1,)
#    f_max = file('minmax.txt', 'a')
#    numpy.savetxt(f_max, maxPix, delimiter=",")
#    numpy.savetxt(f_max, minPix, delimiter=",")
#    f_max.close()
    print parser.name
    print maxPix
    print minPix
    #numpy.savetxt("cyl_proj_area"+filter(str.isdigit,parser.name)+".txt", pixAreaCylProjList, delimiter=",")
    #numpy.savetxt("cyl_proj_flag"+filter(str.isdigit,parser.name)+".txt", pixCode, delimiter=",")
    
    

    """
    
    if verbose:
        griddedPix = 0 
        print('Intersecting pixels')
        sys.stdout.write("Approximately 0 pixels gridded. ")
        sys.stdout.flush()
        pixAreaCylList = []
        pixAreaCylProjList = []
        #pixAreaSinList = []
        #pixMeYList = []
        #pixMeXList = []
        #pixProjYList = []
        #pixProjXList = []
        pixLatList = []
        pixLonList = []
        for (pxlat, pxlon, pxind) in izip(lat, lon, ind):
            if numpy.any(numpy.isnan(pxlat)) or numpy.any(numpy.isnan(pxlon)):
                continue  # if we have only a partial pixel, skip
            if numpy.ptp(pxlon) > 180.:
                continue # if we're going the wrong way around the sphere, skip
            griddedPix += 1
            sys.stdout.write("\rApproximately {0} pixels calculated. ".\
                             format(griddedPix))
            sys.stdout.flush()
            
            pixLatList.append(pxlat)
            pixLonList.append(pxlon)
            
            # find area using Lambert Cylindrical Equal Area projection w/ 0 deg stdpar
            projCyl = numpy.vectorize(utils.cylEqualArea)
            (yCyl, xCyl) = projCyl(pxlat,pxlon)
            pixPolyCyl = geom.MultiPoint(zip(yCyl, xCyl)).convex_hull
            pixAreaCyl = pixPolyCyl.area 
            pixAreaCylList.append(pixAreaCyl)
            #pixMeYList.append(yCyl)
            #pixMeXList.append(xCyl)
            
#            # find area using Sinusoidal projection
#            projSin = numpy.vectorize(utils.sinEqualArea)
#            (ySin, xSin) = projSin(pxlat,pxlon)
#            pixPolySin = geom.MultiPoint(zip(ySin, xSin)).convex_hull
#            pixAreaSin = pixPolySin.area 
#            pixAreaSinList.append(pixAreaSin)
#            
            
            # find using proj version of CEA
            #projCEA = numpy.vectorize(grid_geo.cylequalarea_GridDef.geoToProjected)
            (yCEA, xCEA) = gridCEA.geoToProjected(pxlat,pxlon)
            pixPolyCyl = geom.MultiPoint(zip(yCEA, xCEA)).convex_hull
            pixProjAreaCyl = pixPolyCyl.area 
            pixAreaCylProjList.append(pixProjAreaCyl)
            #pixProjYList.append(yCEA)
            #pixProjXList.append(xCEA)
            
            #num = 0
            
#            if not numpy.allclose(xCEA, xCyl):
#                print "x's do not match"
#                print "Proj: " + str(xCEA)
#                print "My code: " + str(xCyl)
#                num = num+1
#            if not numpy.allclose(yCEA, yCyl):
#                print "y's do not match"
#                num = num+1
                
#            pixRatio = pixAreaCyl/pixAreaSin
#            if pixRatio > 1.01 or pixRatio < 0.99:
#                print "lat = "
#                print pxlat
#                print parser.get('latcorn', indices=pxind)
#                print "lon = "
#                print pxlon
#                print parser.get('loncorn', indices=pxind)
#                print "cylindrical area = " + str(pixAreaCyl)
#                print "sin area = " + str(pixAreaSin)
#                print "ratio = " + str(pixRatio)
        
        print "min pix"
        print numpy.min(pixAreaCylProjList)
        print "max pix"
        print numpy.max(pixAreaCylProjList)
        
#        numpy.savetxt("sin_noedges"+filter(str.isdigit,parser.name)+".txt", pixAreaSinList, delimiter=",")
        numpy.savetxt("cyl_area_"+filter(str.isdigit,parser.name)+".txt", pixAreaCylList, delimiter=",")
        #numpy.savetxt('y_me_'+filter(str.isdigit,parser.name)+".txt", pixMeYList)
        #numpy.savetxt('x_me_'+filter(str.isdigit,parser.name)+".txt", pixMeXList)
        
        numpy.savetxt("cyl_proj_area"+filter(str.isdigit,parser.name)+".txt", pixAreaCylProjList, delimiter=",")
        #numpy.savetxt('y_proj_'+filter(str.isdigit,parser.name)+".txt", pixProjYList)
        #numpy.savetxt('x_proj_'+filter(str.isdigit,parser.name)+".txt", pixProjXList)
    
#        print "Lambert Cylindrical"
#        hist(pixAreaCylList,bins=200)
#        show()
#        numpy.savetxt("lambCyl_0_"+filter(str.isdigit,parser.name)+".txt", histCyl[0], delimiter=",")
#        numpy.savetxt("lambCyl_1_"+filter(str.isdigit,parser.name)+".txt", histCyl[1], delimiter=",")
#        print "Sinusoidal"
#        hist(pixAreaSinList,bins=200)
#        show()
#        numpy.savetxt("sin_0_"+filter(str.isdigit,parser.name)+".txt", histSin[0], delimiter=",")
#        numpy.savetxt("sin_1_"+filter(str.isdigit,parser.name)+".txt", histSin[1], delimiter=",")

    

        # return some stuff with weight=zero so the program runs
        outer_indices = griddef.indLims()
        map = map_helpers.init_output_map(outer_indices)
        areaMap = map_helpers.init_output_map(outer_indices) # omno2d
        if griddedPix != 0: # don't run if nothing gridded
            for (key, areaTup) in areaMap.iteritems(): # loop over gridcells
                for (pxind, areas) in areaTup: # loop over pixels
                    map[key].append((tuple(pxind),0.)) 
    return map
    """
