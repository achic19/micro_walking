## Run when initialise the code
from Code.general_functions import *
import geopandas as gpd
import osmnx as ox
from geopandas import GeoDataFrame, GeoSeries
from osmnx import io

from pandas import DataFrame

project_crs = 'epsg:3857'
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString, MultiPolygon
import math
import warnings
import pandas as pd

from tqdm import tqdm
import sys
sys.setrecursionlimit(5000)  # Set to a higher limit
warnings.filterwarnings(action='ignore')
from momepy import extend_lines

pjr_loc = os.path.dirname(os.getcwd())
from itertools import combinations
import numpy as np
from math import log2

# In this example, the data is extracted from OSM by specifying a location's name The code is designed to handle multiple polygons or location names seamlessly.
# 'San Francisco, California','Turin,Italy'
# Download data from OpenStreetMap, project it, and convert it to a GeoDataFrame. OSMnx automatically resolves topology errors and retrieves only the street-related polylines.
for place in ['Göteborg, Sweden']:

    if place == 'Tel Aviv':
        useful_tags_path = ['name:en', 'highway', 'length', 'bearing', 'tunnel', 'junction']
        ox.utils.config(useful_tags_way=useful_tags_path)
    my_preprocessing = Preprocessing(place)
    data_folder = my_preprocessing.create_folder()
    graph = ox.graph_from_place(place, network_type='all')
    print('finish to download data')
    graph = ox.bearing.add_edge_bearings(graph, precision=1)
    graph_pro = ox.projection.project_graph(graph, to_crs=project_crs)
    io.save_graph_geopackage(graph_pro, filepath=f'{data_folder}/osm_data.gpkg', encoding='utf-8', directed=False)
    df_pro = my_preprocessing.first_filtering()
    df_pro.to_file(f'{data_folder}/before_df.shp')
    print('calculate simplification')


    # Functions and classes to be utilized - Module 2
    def check_parallelism(to_translate: GeoDataFrame) -> bool:
        # See if there are parallel lines
        my_buffer = to_translate['geometry'].buffer(cap_style=2, distance=30, join_style=3)
        to_translate['geometry_right'] = to_translate['geometry'].apply(lambda x: x.parallel_offset(35, 'right'))
        to_translate['geometry_left'] = to_translate['geometry'].apply(lambda x: x.parallel_offset(35,
                                                                                                   'left'))  # we need to offset by both sides since the parallel lines could be in opposite directions

        def is_parallel(my_s_join: GeoDataFrame, the_buffer: GeoSeries, geo_field: str):
            my_s_join['geometry'] = my_s_join[geo_field]
            new_data_0 = my_s_join.sjoin(GeoDataFrame(geometry=the_buffer, crs=project_crs), how='inner').reset_index()
            if not len(new_data_0):
                return False
            new_data_1 = new_data_0[
                new_data_0['index'] != new_data_0['index_right']]  # Remove overlay of polylines with its buffer
            for translated_line in new_data_1.iterrows():
                translated_line = translated_line[1]
                geo_tr_line = GeoDataFrame(data=pd.DataFrame([translated_line]), crs=project_crs)
                overlay = gpd.overlay(geo_tr_line, GeoDataFrame(geometry=the_buffer.loc[geo_tr_line['index_right']],
                                                                crs=project_crs), how='intersection')
                if (overlay.length / translated_line.length).iloc[0] * 100 > 10:
                    return True
            return False

        if is_parallel(to_translate, my_buffer, 'geometry_right'):
            return True
        else:
            if is_parallel(to_translate, my_buffer, 'geometry_left'):
                return True
            else:
                return False


    def create_center_line(one_poly):
        """
        This method calculate new line between the farthest points of the simplified polygon
        :param one_poly:
        :return:
        """
        lines_in_buffer = data.sjoin(GeoDataFrame(geometry=[one_poly], crs=project_crs)).drop(columns='index_right')

        list_pnts_of_line_group = []

        def update_list(line_local):
            """
            add the first start/end point into the list
            :param line_local:
            :return:
            """
            list_pnts_of_line_group.extend([Point(line_local.coords[0]), Point(line_local.coords[-1])])

        # Get the start/end points of these polylines
        lines_in_buffer['geometry'].apply(update_list)

        # Find all the unidirectional combinations between each two pair of points
        point_combinations = list(combinations(list_pnts_of_line_group, 2))

        # Save it into DataFrame frame and calculate distance
        df_test = DataFrame()
        df_test['point_1'] = [pair[0] for pair in point_combinations]
        df_test['point_2'] = [pair[1] for pair in point_combinations]
        df_test['dist'] = df_test.apply(lambda x: x['point_1'].distance(x['point_2']), axis=1)

        # Calculate  angle (0 and 180)
        # Calculate angle using vectorized operations
        # Vectorized angle calculation using NumPy
        dx = df_test['point_2'].apply(lambda p: p.x) - df_test['point_1'].apply(lambda p: p.x)
        dy = df_test['point_2'].apply(lambda p: p.y) - df_test['point_1'].apply(lambda p: p.y)
        df_test['angle'] = np.degrees(np.arctan2(dy, dx))
        df_test['angle'] = np.where(df_test['angle'] > 0, df_test['angle'], df_test['angle'] + 180)

        # Calculate the best two points by looking on their distance and angle. we compare the angle to the polylines angles. The angle has less important so the reason for 0.5
        avg = lines_in_buffer['angle'].mean()
        dis = abs(df_test['angle'] - avg)
        df_test['ratio'] = df_test['dist'] / df_test['dist'].max() + 0.5 * dis / dis.max()
        max_points = df_test.sort_values(by='ratio', ascending=False).iloc[0]

        # These points will be served to be initial reference in order to find more points
        pnt_f = max_points['point_1']
        pnt_l = max_points['point_2']

        angl_rng = lines_in_buffer['angle'].max() - lines_in_buffer['angle'].min()
        if angl_rng < 1:  # If the angel range is less than 1 degree the line will be based on the first and last points
            lines_pnt_geo = [pnt_f]
        else:
            if angl_rng > 100:  # Maximum of length to check is every 10 meters
                length_to_check = 8.561438102
            else:
                length_to_check = 75 - log2(
                    angl_rng) * 10  # The range of  length_to_check (logarithm to create more changes at the beginning)
            lines_pnt_geo = add_more_pnts_to_new_lines(pnt_f, pnt_l, [pnt_f], length_to_check, lines_in_buffer)
        lines_pnt_geo.append(pnt_l)
        # Update dic_final
        return lines_pnt_geo


    def add_more_pnts_to_new_lines(pnt_f_loc: Point, pnt_l_loc: Point, line_pnts: list, lngth_chck: float,
                                   test_poly: GeoDataFrame) -> list:
        """
        This method checks if more points should be added to the new lines by checking along the new line if the distance to the old network roads are more than 10 meters
        :param test_poly: From these polylines find the closet one in each interation
        :param lngth_chck: Used latter to find how many checks should be done
        :return:
        """
        # Calculate distance and azimuth between the first and last point
        dist = pnt_f_loc.distance(pnt_l_loc)
        x_0 = pnt_f_loc.coords[0][0]
        y_0 = pnt_f_loc.coords[0][1]
        bearing = math.atan2(pnt_l_loc.coords[0][0] - x_0, pnt_l_loc.coords[0][1] - y_0)
        bearing = bearing + 2 * math.pi if bearing < 0 else bearing
        # Calculate the number of  checks going to carry out
        loops = int(dist / lngth_chck)
        # Calculate  the first point over the line
        for dis_on_line in range(1, loops):
            x_new = x_0 + lngth_chck * dis_on_line * math.sin(bearing)
            y_new = y_0 + lngth_chck * dis_on_line * math.cos(bearing)
            # S_joins to all the network lines (same name and group)
            # if the distance is less than 10 meters continue, else: find the projection point and add it to the correct location and run the function agein
            one_pnt_df = GeoDataFrame(geometry=[Point(x_new, y_new)], crs=project_crs)
            s_join_loc = one_pnt_df.sjoin_nearest(test_poly, distance_col='dis').iloc[0]

            if s_join_loc['dis'] > 10:
                line = data.loc[s_join_loc['index_right']]['geometry']
                pnt_med = line.interpolate(line.project(s_join_loc['geometry']))
                if pnt_med.distance(pnt_f_loc) < 10:  # Otherwise the code may stack in endless loops
                    continue
                line_pnts.append(pnt_med)
                line_pnts = add_more_pnts_to_new_lines(pnt_med, pnt_l_loc, line_pnts, lngth_chck, test_poly)
                return line_pnts
        return line_pnts


    def update_df_with_center_line(new_line, is_simplified=0, group_name=-1):
        """
        update our dictionary with new lines
        :param is_simplified:
        :param new_line:
        :param group_name: According to the DBSCAN algorithm, if no =-1
        :return:
        """
        dic_final['name'].append(name)
        # dic_final['geometry'].append(LineString(coordinates=(pnt_list[max_dis[0]], pnt_list[max_dis[1]])))
        dic_final['geometry'].append(new_line)
        dic_final['highway'].append(data.iloc[0]['highway'])  # ToDo should be fixed to take the largest one
        dic_final['bearing'].append(data['angle'].mean())
        dic_final['group'].append(group_name)
        dic_final['is_simplified'].append(is_simplified)


    # Function to calculate circular_distance
    def circular_distance(angle1, angle2):
        diff = np.abs(angle1 - angle2) % 180
        return np.minimum(diff, 180 - diff)


    # Initiate dic_final here for @def update_df_with_center_line
    dic_final = {'name': [], 'geometry': [], 'highway': [], 'bearing': [], 'group': [], 'is_simplified': []}

    # group the street segments by street name
    my_groupby = df_pro.groupby('name')
    for_time = len(my_groupby)
    number_of_parallel = 0  # count the number of polylines were refined
    with tqdm(total=for_time) as pbar:  # It is used in order to visualise the progress by progress bar
        for i, street in enumerate(my_groupby):
            res = street[1]  # it holds all the streets
            name = street[0]  # It holds the streets name
            print(name)
            pbar.update(1)  # for the progress bar
            # Remove segments without angle. If less than two segments being left move to the next group.
            res = res.dropna(subset=['angle'], axis=0)
            if len(res) < 2:
                data = res
                _ = res['geometry'].apply(lambda x: update_df_with_center_line(x))
                continue
            # Use DBSCAN to classify streets based on their angle, and group each class. Outliers could not consider parallel with any street, thus removed
            angles = res['angle'].to_numpy()
            # Compute pairwise distances between angles
            pairwise_distances = np.zeros((len(angles), len(angles)))
            for i in range(len(angles)):
                for j in range(len(angles)):
                    pairwise_distances[i, j] = circular_distance(angles[i], angles[j])
            # Use DBSCAN
            epsilon = 10
            min_samples = 2  # Adjust as needed
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
            res['group'] = dbscan.fit_predict(pairwise_distances)
            # if all is -1, don't touch the element
            if (res['group'] == -1).all():
                data = res
                _ = res['geometry'].apply(lambda x: update_df_with_center_line(x))
                continue
            # cur_group = res[(res['group'] > -1) | (res.length>20)].groupby('group') # Remove short segments with -1 classification values
            # The parallel test is on street segments that  have the same name and belong to the same angle group.
            for group in res.groupby('group'):
                data = group[1]
                if group[0] == -1:  # No need to check if is parallel
                    _ = data['geometry'].apply(lambda x: update_df_with_center_line(x))
                    continue
                if check_parallelism(data.copy()):
                    # print(group[0])
                    # Remove unimportant streets which appear more than 10% in the group
                    min_num_of_polylines = len(data) / 15
                    # Use a single boolean condition for filtering
                    condition = (data['highway'].isin(['service', 'unclassified'])) & (
                            data.groupby('highway')['highway'].transform('count') <= min_num_of_polylines)
                    data = data[~condition]

                    number_of_parallel += len(data)  # Update the number of parallel polylines

                    # unify lines to one polygon
                    buffers = data.buffer(cap_style=3, distance=30, join_style=3)
                    one_buffer = buffers.unary_union
                    # simplify polygon with simplify function. If one_buffer is multipolygon object simplify each one them separately
                    if isinstance(one_buffer, MultiPolygon):
                        for polygon in one_buffer.geoms:
                            lines_pnt_geo_final = create_center_line(polygon)
                            update_df_with_center_line(LineString(lines_pnt_geo_final), 1, group[0])
                    else:
                        lines_pnt_geo_final = create_center_line(one_buffer)
                        # Update dic_final
                        update_df_with_center_line(LineString(lines_pnt_geo_final), 1, group[0])

                else:
                    _ = data['geometry'].apply(lambda x: update_df_with_center_line(x))

    print(f'number_of_parallel: {number_of_parallel}')
    print('create new files')
    # remove short lines
    final_cols = ['name', 'geometry', 'highway', 'bearing', 'length']
    new_network = GeoDataFrame(dic_final, crs=project_crs)
    new_network['length'] = new_network.length
    # create network
    new_network.to_file(f'{data_folder}/simp.shp')

    num = 0
    new_gpd = new_network.copy()
    obj_intersection = Intersection(new_gpd, num)
    obj_intersection.intersection_network()
    obj_intersection.update_names(new_gpd)

    line_name = 'line_name'
    if my_preprocessing.is_junction:
        print('Update roundabout')
        exist_data = obj_intersection.my_network.reset_index().reset_index(names=line_name)
        my_roundabout = Roundabout(exist_data, my_preprocessing.round_about)
        deadend_lines, deadend_pnts = my_roundabout.deadend()

        # update the current network
        change_geo = my_roundabout.my_spatial_join(deadend_lines, deadend_pnts, line_name)
        my_roundabout.update_the_current_network(change_geo)

        my_roundabout.network.drop_duplicates(subset=line_name, inplace=True)
        # Improve roundabout
        # First buffer around centroid
        centr_name = 'centr_name'
        buffer_around_centroid = my_roundabout.centroid['geometry'].buffer(cap_style=1, distance=30)

        # s_join between buffer to lines (reset index to retain the original centroid name which can apper more than one in the results). always stay with data you need and with understandable name
        roundabout_with_lines = \
            gpd.sjoin(left_df=GeoDataFrame(geometry=buffer_around_centroid, crs=project_crs).reset_index(),
                      right_df=my_roundabout.network[['geometry', line_name]]).drop_duplicates(
                subset=['index', line_name]).rename(columns={"index": centr_name})[['geometry', line_name, centr_name]]

        # To facilitate the searching process
        my_roundabout.network.set_index(line_name, inplace=True)
        # To facilitate easy access to point centroid geometry data, it is advisable to store the information in an object that provides efficient retrieval.
        pnt_centroid_temp = my_roundabout.centroid['geometry']
        #  Group the data by centroid
        for center_line in roundabout_with_lines.groupby(centr_name):
            #  Iterate over each group after performing a groupby() operation
            for center in center_line[1].itertuples():
                # Find the line that connects to the current centroid and obtain its vertices
                line_to_test = my_roundabout.network.loc[center[2]]
                vertices_line = list(line_to_test['geometry'].coords)
                pnt_test = [vertices_line[0], vertices_line[-1]]
                # To determine if the current line is already connected to the current centroid,.
                is_connected = my_roundabout.centroid[
                    my_roundabout.centroid['geometry'].isin([Point(pnt_test[0]), Point(pnt_test[-1])])]
                if len(is_connected) > 0 and center[3] in is_connected['name']:
                    continue

                if len(vertices_line) == 2:
                    vertices_line.insert(1, pnt_centroid_temp[center[3]])
                else:
                    my_list = [pnt_centroid_temp[center[3]].distance(Point(temp)) for temp in vertices_line]
                    # Find the minimum index
                    min_index = min(range(len(my_list)), key=my_list.__getitem__)
                    if min_index == 0:
                        vertices_line.insert(0, pnt_centroid_temp[center[3]])
                    elif min_index == len(my_list) - 1:
                        vertices_line.append(pnt_centroid_temp[center[3]])
                    else:
                        vertices_line[min_index] = pnt_centroid_temp[center[3]]
                new_geo = LineString(vertices_line)
                my_roundabout.network.at[center[2], 'geometry'] = new_geo

        new_network1 = my_roundabout.network.reset_index()
        new_network1.drop(columns='index', inplace=True)


        # Function to remove self-intersecting LineString geometries
        def remove_self_intersecting(line):
            return line.is_simple


        # Apply the function to filter out self-intersecting geometries
        new_network2 = new_network1[new_network1['geometry'].apply(remove_self_intersecting)]
    else:
        new_network2 = obj_intersection.my_network.reset_index()
    extend_lines_f = extend_lines(new_network2, 100)
    extend_lines_f['length'] = extend_lines_f.length

    obj_intersection_1 = Intersection(extend_lines_f.copy(), 1)
    obj_intersection_1.intersection_network()
    obj_intersection_1.my_network.rename(columns={'str_name': 'name'},
                                         inplace=True)  # 'str_name' become 'str to be compatible with other previous networks
    obj_intersection_1.update_names(org_gpd=extend_lines_f.copy().rename(columns={'str_name': 'name'}))

    # Clear short segments
    final2 = EnvEntity(obj_intersection_1.my_network.reset_index())
    final2.update_the_current_network(final2.get_deadend_gdf(delete_short=100))
    final2.network.drop(columns=['bearing']).to_file(f'{data_folder}/network.shp')

    # Aggregation
    print('Aggregate intersections')
    network = final2.network

    # 1. Get the first/start of each line
    # Extract unique start and end points from all LineStrings
    geometry = 'geometry'
    index_right = 'index_right'
    all_points = network[geometry].apply(lambda line: [Point(line.coords[0]), Point(line.coords[-1])]).explode()
    # # Create a GeoSeries of unique points
    unique_points = GeoDataFrame(geometry=gpd.GeoSeries(all_points).unique(), crs=project_crs)
    # save data

    # 2. Make sure I have the name of the lines associated with these lines
    pnts_line_name = unique_points.sjoin(network)[[index_right, geometry]].reset_index().dissolve(by='index',
                                                                                                  aggfunc=lambda
                                                                                                      x: x.tolist())
    pnts_line_name['num_of_lines'] = pnts_line_name[index_right].apply(len)  # count the number of lines for each point

    # 3. Use DBSCAN with 20 meters threshold
    # Extract coordinates for DBSCAN
    coordinates = pnts_line_name.geometry.apply(lambda point: (point.x, point.y)).tolist()
    dbscan = DBSCAN(eps=20, min_samples=2)
    pnts_line_name['group'] = dbscan.fit_predict(coordinates)
    lines_to_update = pnts_line_name[pnts_line_name['group'] > -1]


    # if you want to save the files
    def save_points_file(data, path):
        """
        The function get a data, arrange columns, convert list to string and export  it into a shpfile
        :param data:
        :param path:
        :return:
        """
        col_of_lists_as_str = 'col_of_lists_as_str'
        data[col_of_lists_as_str] = data[index_right].apply(str)
        data.drop(columns=[index_right]).to_file(path)
        data.drop(columns=[col_of_lists_as_str], inplace=True)


    # 4.1.Find the point with the max number of connected lines, if it is one use it otherwise uses the average
    # Find the maximum 'num' value for each group
    num = 'num_of_lines'
    group_name = 'group'
    new_geometry = 'new_geometry'
    max_values_per_group = lines_to_update.groupby('group')['num_of_lines'].max()
    # Filter rows with the maximum 'num' value for each group
    result_gdf = lines_to_update[
        lines_to_update.set_index([group_name, num]).index.isin(list(max_values_per_group.items()))]


    # Custom aggregation function to calculate the average point for each group
    def calculate_average_point(group):
        x_mean = group.x.mean()
        y_mean = group.y.mean()
        return Point(x_mean, y_mean)


    # Apply the custom aggregation function to calculate average points per group
    lines_to_update2 = lines_to_update.set_index(group_name)
    lines_to_update2['new_geometry'] = result_gdf.groupby(group_name)[geometry].apply(calculate_average_point)

    # 4.2 Among whom are updated remove every line the start and last point are the same
    # Get all the lines going to be deleted
    lines_to_delete = []


    def update_lines_to_delete(row):
        # explode the lines names within each row list to separate rows
        lines_to_update_tmep = row[index_right].explode()

        # Identify rows with duplicate values
        lines_to_delete.extend(lines_to_update_tmep[lines_to_update_tmep.duplicated()].tolist())


    lines_to_update2.groupby(level=group_name).apply(update_lines_to_delete)

    # remove lines their geometry not going to change
    lines_to_update3 = lines_to_update2[lines_to_update2[geometry] != lines_to_update2[new_geometry]]

    # 4.3 Change the point of each line with new point
    network_new = network[~network.index.isin(lines_to_delete)]


    def update_network_with_aggregated_point(group):
        lines_in_group = group.explode(index_right)

        def update_one_line(points_data):
            if points_data.name not in lines_to_delete:
                updated_line_geo = network_new.loc[points_data.name]
                line_coords = updated_line_geo.geometry.coords
                if Point(line_coords[0]) == points_data.geometry:
                    network_new.at[points_data.name, geometry] = LineString(
                        [points_data[new_geometry]] + line_coords[1:])
                elif Point(line_coords[-1]) == points_data.geometry:
                    network_new.at[points_data.name, geometry] = LineString(
                        line_coords[:-1] + [points_data[new_geometry]])
                else:
                    print(points_data)
                    print(lines_in_group)

        lines_in_group.set_index(index_right).apply(update_one_line, axis=1)


    lines_to_update3.groupby(level=group_name).apply(update_network_with_aggregated_point)
    network_new['length'] = network_new.length
    network_new.drop(columns=[ 'bearing']).to_file(f'{data_folder}/network_new.shp')
