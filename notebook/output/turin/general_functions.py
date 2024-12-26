import os
import warnings
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, LineString, MultiPolygon, MultiPoint
import osmnx as ox
import ast
from momepy import remove_false_nodes, extend_lines
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings(action='ignore')

# Set project coordinate reference system
project_crs = 'epsg:3857'

# Get the current working directory
pjr_loc = os.getcwd()


class Preprocessing:
    def __init__(self, place, is_test=False):
        """
        Initialize the Preprocessing class with a place name.

        Parameters:
        place (str): Name of the place to be processed.
        is_test (bool): Flag to indicate if this is a test run.
        """
        self.place = place
        self.round_about = None  # To store elements of roundabout type
        self.is_junction = True  # True if the dataset includes the 'junction' column after downloading from OSM
        self.data_folder = None  # Folder to store all the new shapefiles
        self.is_test = is_test

    def create_folder(self):
        """
        Create necessary folders for the place.

        Parameters:
        is_test (bool): If True, creates test folders.

        Returns:
        str: Path to the data folder.
        """
        folder_name = self.place.replace(",", "_").replace(" ", "_")
        self.data_folder = os.path.join(os.path.dirname(pjr_loc), f'places/{folder_name}')
        os.makedirs(self.data_folder, exist_ok=True)
        print(f'Data folder: {self.data_folder}')
        if self.is_test:
            self.data_test = os.path.join(os.path.dirname(pjr_loc), f'places/{folder_name}/test')
            os.makedirs(os.path.join(self.data_test, 'topology'), exist_ok=True)
            os.makedirs(os.path.join(self.data_test, 'preprocessing'), exist_ok=True)
            os.makedirs(os.path.join(self.data_test, 'simplification'), exist_ok=True)
            print(f'Data folder: {self.data_test}')
            return self.data_folder, self.data_test
        else:
            return self.data_folder

    def first_filtering(self):
        """
        Perform the first filtering of polylines based on certain criteria.

        Returns:
        GeoDataFrame: A filtered GeoDataFrame.
        """
        # Read the GeoDataFrame from the specified GeoPackage layer
        my_gdf = gpd.read_file(os.path.join(self.data_folder, 'osm_data.gpkg'), layer='edges')

        # Handle specific case for 'Tel Aviv'
        if self.place == 'Tel Aviv':
            my_gdf.rename(columns={'name:en': 'name'}, inplace=True)

        def remove_unconnected_streets(df_connected, str_name='name_left', con_str_name='name_right'):
            """
            Remove streets from a GeoDataFrame that are not connected to any other streets.

            Parameters:
            df_connected (GeoDataFrame): The input GeoDataFrame containing street data.
            str_name (str): The column name for the street name in the first dataset.
            con_str_name (str): The column name for the connected street name in the second dataset.

            Returns:
            GeoDataFrame: A new GeoDataFrame with unconnected streets removed.
            """
            df_analysis = df_connected.copy()
            s_join_analysis = gpd.sjoin(df_analysis, df_connected)
            s_join_analysis2 = s_join_analysis[s_join_analysis[str_name] != s_join_analysis[con_str_name]]
            not_connected = set(df_connected['name']) - set(s_join_analysis2.reset_index()[str_name])
            df_pro_filtered = df_connected[~df_connected['name'].isin(not_connected)]
            return df_pro_filtered

        # Check if 'junction' column exists
        self.is_junction = 'junction' in my_gdf.columns
        if self.is_junction:
            self.round_about = my_gdf[my_gdf['junction'].isin(['roundabout', 'circular'])]
            my_gdf = remove_unconnected_streets(my_gdf)
            my_gdf = my_gdf[~((my_gdf['junction'] == 'roundabout') | (my_gdf['junction'] == 'circular'))]

        # Filter out specific highway types
        to_remove = my_gdf[~my_gdf['highway'].isin(['motorway', 'trunk', 'motorway_link', 'trunk_link'])]
        df_pro = to_remove.to_crs(project_crs).dropna(subset=['name'])
        df_pro = df_pro[df_pro['name'] != '']
        df_pro['angle'] = df_pro['bearing'].apply(lambda x: x if x < 180 else x - 180)
        df_pro['length'] = df_pro.length

        # Function to convert valid list strings to lists
        def convert_to_list(s):
            try:
                return ast.literal_eval(s)[0]
            except (ValueError, SyntaxError, TypeError):
                return s

        df_pro['name'] = df_pro['name'].apply(convert_to_list)
        df_pro['highway'] = df_pro['highway'].apply(convert_to_list)
        if self.is_test:
            return df_pro,self.round_about
        else:
            return df_pro

class Simplification:
    pass
# Classes to be employed during the execution of this code.
# Intersection
# Split in intersection
class Intersection:
    def __init__(self, network: GeoDataFrame, number: int):
        """

        :param network:
        :param number: give a unique name to the files created during the process (this class will be use again in this code)
        """
        self.my_network = network
        self.inter_pnt_dic = {'geometry': [], 'name': []}
        self.lines_to_delete = []
        self.num = number
        print('Update topology')

    def intersection_network(self):
        """
        This function fix topology (add or remove vertices) where needed
        :return:
        """
        # First remove_false_nodes
        self.my_network = remove_false_nodes(self.my_network).reset_index(drop=True)
        # Create buffer around each element
        buffer_around_lines = self.my_network['geometry'].buffer(cap_style=3, distance=1, join_style=3)

        # s_join between buffer to lines
        s_join_0 = gpd.sjoin(left_df=GeoDataFrame(geometry=buffer_around_lines, crs=project_crs),
                             right_df=self.my_network)

        # delete lines belong to the buffer
        s_join = s_join_0[s_join_0.index != s_join_0['index_right']]

        # Find new intersections that are not at the beginning or end of the line
        for_time = len(s_join)
        with tqdm(total=for_time) as pbar:
            s_join.apply(lambda x: self.find_intersection_points(x, pbar), axis=1)
        if len(self.inter_pnt_dic) == 0:
            return
        inter_pnt_gdf = GeoDataFrame(self.inter_pnt_dic, crs=project_crs)
        # Split string line by points
        segments = {'geometry': [], 'org_id': []}
        # Groupby points name (which is the line they should split)
        my_groups = inter_pnt_gdf.groupby('name')
        for_time = len(my_groups)
        with  tqdm(total=for_time) as pbar:
            for group_pnts in my_groups:
                pbar.update(1)
                points = group_pnts[1]
                points['is_split'] = True
                # if group_pnts[0]==588:
                #     print(points)
                # get the line to split by comparing the name
                row = self.my_network.loc[group_pnts[0]]
                current = list(row.geometry.coords)
                points_line = [Point(x) for x in current]
                points_line_gdf = GeoDataFrame(geometry=points_line, crs=project_crs)
                points_line_gdf['is_split'] = False

                # append all the points together (line points and split points)
                line_all_pnts = GeoDataFrame(pd.concat([points_line_gdf, points]), crs=project_crs)

                # Find the distance of each point form the begining of the line on the line.
                line_all_pnts['dis_from_the_start'] = line_all_pnts['geometry'].apply(
                    lambda x: row.geometry.project(x))
                line_all_pnts.sort_values('dis_from_the_start', inplace=True)

                # split the line
                seg = []
                for point in line_all_pnts.iterrows():
                    prop = point[1]
                    seg.append(prop['geometry'])
                    if prop['is_split']:
                        segments['geometry'].append(LineString(seg))
                        segments['org_id'].append(row.name)
                        seg = [prop['geometry']]
                # if the split point is the last one, you don't need to create new segment
                if len(seg) > 1:
                    segments['geometry'].append(LineString(seg))
                    segments['org_id'].append(row.name)
        network_split = GeoDataFrame(data=segments, crs=project_crs)
        cols_no_geometry = self.my_network.columns[:-1]
        network_split_final = network_split.set_index('org_id')
        network_split_final[cols_no_geometry] = self.my_network[cols_no_geometry]
        # remove old and redundant line from our network and update with new one
        network_split = GeoDataFrame(pd.concat([self.my_network.drop(index=network_split_final.index.unique()),
                                                network_split_final]), crs=project_crs)
        network_split['length'] = network_split.length
        self.my_network = network_split
        self.my_network.reset_index(drop=True, inplace=True)

    def find_intersection_points(self, row, pbar):
        r"""
        find the intersection points between the two lines
        :param row:
        :return:
        """
        pbar.update(1)
        line_1 = self.my_network.loc[row.name]
        line_2 = self.my_network.loc[row['index_right']]
        pnt = line_1.geometry.intersection(line_2.geometry)
        try:
            # If there are more than one intersection between two lines, one of the lines should be deleted.
            if isinstance(pnt,
                          LineString):  # The intersection is only between the buffer and the point
                return
            if isinstance(pnt, MultiPoint):
                for single_pnt in pnt.geoms:
                    self.inter_pnt_dic['geometry'].append(single_pnt)
                    self.inter_pnt_dic['name'].append(row.name)
                return
            # If it is first or end continue OR if there is no intersection between the two lines
            if len(pnt.coords) == 0 or pnt.coords[0] == line_1.geometry.coords[0] or pnt.coords[0] == \
                    line_1.geometry.coords[-1]:
                return
            self.inter_pnt_dic['geometry'].append(pnt)
            self.inter_pnt_dic['name'].append(row.name)
        except:
            print(f"{row.name},{row['index_right']}:{pnt}")

    def update_names(self, org_gpd: GeoDataFrame):
        """
        It updates the name of those lost their name during the previous process
        :param org_gpd:
        :return:
        """
        df1 = self.my_network
        # Split df1 into two GeoDataFrames: df3 (with names) and df4 (without names)
        df3 = df1[df1['name'].notna()]
        df4 = df1[df1['name'].isna()]

        # use only one polyline from the original dataframe for name even if the algorithm may found more
        old_index = 'old_index'
        df4_as_buffer = GeoDataFrame(geometry=df4['geometry'].buffer(distance=2, cap_style=2), crs=project_crs)
        df = gpd.sjoin(df4_as_buffer,
                       org_gpd)  # for spatial join use buffer around each polyline.that provide better result
        df.index.name = old_index
        df['geometry'] = df4['geometry']  # bring the dataframe into linestring format
        df.reset_index(inplace=True)  # To be consistent with the following code and other dataframe
        # Create a new dictionary to store the updated data.
        dic_str_data = []

        def return_street_name(aplcnts_tst):
            """
            1. "Count the occurrences of polylines with the same name within each aplcnts_tst."
            2. "Return the street if a aplcnts_tst contains only one unique street name."
            3. "If a single street name predominates within a aplcnts_tst, return that name."
            4. "For groups with multiple names, perform a buffer calculation around the respective polylines and determine the largest overlapping area, returning the name associated with that area."
            :param aplcnts_tst: group of applicants. Some of them hold the correct street name
            :return:
            """
            count_names = aplcnts_tst['name'].value_counts().sort_values(ascending=False)
            if len(count_names) == 1:
                # there is only one name
                my_data = aplcnts_tst.iloc[0]
            elif count_names[1] - count_names[0] > 1:
                # The highest number of polylines with the same name are bigger at least in 2:
                my_data = aplcnts_tst[aplcnts_tst['name'] == count_names.index[0]].iloc[0]
            else:
                # otherwise filter those with the most popular name or close to (-1)
                str_to_wrk_on = aplcnts_tst[
                    aplcnts_tst['name'].isin(count_names[count_names - count_names[0] < 2].index)]
                buffer_0 = GeoDataFrame(
                    geometry=[str_to_wrk_on.iloc[0]['geometry'].buffer(distance=20, cap_style=2)],
                    crs=project_crs)  # Buffer around the polyline without name

                streets_right_geo = org_gpd[org_gpd.index.isin(str_to_wrk_on[
                                                                   'index_right'])].reset_index()  # Get all the applicants polylines and create buffer around
                buffer_1 = GeoDataFrame(geometry=streets_right_geo.buffer(distance=20, cap_style=2))
                streets_right_geo['area'] = gpd.overlay(buffer_1, buffer_0, how='intersection').area
                groupy = streets_right_geo.groupby('name')
                my_data_0 = \
                    groupy.get_group(groupy['area'].sum().sort_values(ascending=False).index[0]).sort_values(
                        by='area',
                        ascending=False).iloc[
                        0]
                # Get back to the @aplcnts_tst and find the relevant row by comparing index
                my_data = aplcnts_tst[aplcnts_tst['index_right'] == my_data_0['index']].iloc[0]
            # Populate the new dictionary with relevant data
            dic_str_data.append(my_data.to_list())

        _ = df.groupby(old_index).apply(return_street_name)
        # convert the dictionary into a dataframe.
        updated_df = GeoDataFrame(data=dic_str_data, columns=df.columns, crs=project_crs).drop(
            columns='index_right').set_index(old_index)
        updated_df['length'] = updated_df.length
        self.my_network = GeoDataFrame(pd.concat([df3, updated_df]), crs=project_crs)


# Roundabout
class EnvEntity:
    def __init__(self, network):
        self.dead_end_fd = None
        self.pnt_dead_end = None
        self.pnt_dic = {}
        self.first_last_dic = {'geometry': [], 'line_name': [], 'position': []}
        self.network = network

    def __populate_pnt_dic(self, point: type, name_of_line: str):
        """
            Make "pnt_dic" contain a list of all the lines connected to each point.
            :param point:
            :param name_of_line:
            :return:
            """
        if not point in self.pnt_dic:
            self.pnt_dic[point] = []
        self.pnt_dic[point].append(name_of_line)

    def __send_pnts(self, temp_line: GeoSeries):
        """
            # Send the first and the last points to populate_pnt_dic
            :return:
            """
        my_geom = temp_line['geometry']
        self.__populate_pnt_dic(my_geom.coords[0], temp_line.name)
        self.__populate_pnt_dic(my_geom.coords[-1], temp_line.name)

    def get_deadend_gdf(self, delete_short: int = 30) -> GeoDataFrame:
        self.network.apply(self.__send_pnts, axis=1)

        deadend_list = [item[1][0] for item in self.pnt_dic.items() if len(item[1]) == 1]
        pnt_dead_end_0 = [item for item in self.pnt_dic.items() if
                          len(item[1]) == 1]  # Retain all the line points with deadened
        self.pnt_dead_end = [Point(x[0]) for x in pnt_dead_end_0]
        # Create shp file of deadened_pnts
        geometry, line_name = 'geometry', 'line_name'
        pnt_dead_end_df = GeoDataFrame(data=pnt_dead_end_0)
        pnt_dead_end_df[geometry] = pnt_dead_end_df[0].apply(lambda x: Point(x))
        pnt_dead_end_df[line_name] = pnt_dead_end_df[1].apply(lambda x: x[0])
        pnt_dead_end_df.crs = project_crs
        self.dead_end_fd = pnt_dead_end_df

        if delete_short > 0:
            # If it is necessary to eliminate dead-end short segments, it is  important to delete them from the network geodataframe.

            deadend_gdf = self.network.loc[deadend_list]
            self.network.drop(index=deadend_gdf[deadend_gdf.length < delete_short].index, inplace=True)
            return deadend_gdf[deadend_gdf.length > delete_short]
        return self.network.loc[deadend_list]

    def update_the_current_network(self, temp_network):
        r"""
            Update the current network in the new changes
            :param temp_network:
            :return:
            """
        new_network_temp = self.network.drop(index=temp_network.index)
        self.network = GeoDataFrame(pd.concat([new_network_temp, temp_network]), crs=project_crs)
        self.network['length'] = self.network.length
        self.network = self.network[self.network['length'] > 1]


class Roundabout(EnvEntity):
    def __init__(self, network: GeoDataFrame, roundabout_as_poly: GeoDataFrame):
        EnvEntity.__init__(self, network)
        self.pnt_dic = {}
        self.centroid = self.__from_roundabout_to_centroid(roundabout_as_poly)
        self.network.rename(columns={'name': 'str_name'}, inplace=True)

    def __from_roundabout_to_centroid(self, roundabout_file):
        # Find the center of each roundabout
        # create polygon around each polygon and union
        round_about_buffer = roundabout_file.to_crs(project_crs)['geometry'].buffer(cap_style=1,
                                                                                    distance=10,
                                                                                    join_style=1).unary_union
        dic_data = {'name': [], 'geometry': []}
        if round_about_buffer.type == 'Polygon':  # In case we have only one polygon
            dic_data['name'].append(0)
            dic_data['geometry'].append(round_about_buffer.centroid)
        else:
            for ii, xx in enumerate(round_about_buffer.geoms):
                dic_data['name'].append(ii)
                dic_data['geometry'].append(xx.centroid)
        centroid = GeoDataFrame(dic_data, crs=project_crs)
        return centroid

    def __first_last_pnt_of_line(self, row: GeoSeries):
        r"""
        It get geometry of line and fill the first_last_dic with the first and last point and the name of the line
        :return:
        """
        geo = list(row['geometry'].coords)
        self.first_last_dic['geometry'].extend([Point(geo[0]), Point(geo[-1])])
        self.first_last_dic['line_name'].extend([row.name] * 2)
        self.first_last_dic['position'].extend([0, -1])

    def deadend(self):
        r"""
        remove not connected line shorter than 100 meters and then return deadend_list lines and their endpoints (as another file)
        :return:
        """
        # Find the first and last points

        # Get deadend_gdf
        deadend_gdf = self.get_deadend_gdf()

        # Create gdf of line points with the reference to the line they belong
        deadend_gdf.apply(self.__first_last_pnt_of_line, axis=1)
        first_last_gdf = GeoDataFrame(self.first_last_dic, crs=project_crs)

        return deadend_gdf, first_last_gdf

    def __update_geometry(self, cur, s_join):
        r"""
        :return:
        """
        if cur['highway'] == 'footway':
            # Don't snap footway to roundabout
            return cur['geometry']
        # Get only the points that are deadened
        points_lines = [item for item in s_join[s_join['line_name'] == cur.name].iterrows() if
                        item[1]['geometry'] in self.pnt_dead_end]
        if len(points_lines) == 0:
            # No roundabout nearby
            return cur['geometry']
        # get the line geometry to change the first and/ or last point
        geo_cur = list(cur['geometry'].coords)

        # iterate over the deadened points  near roundabout
        for ind in range(len(points_lines)):
            points_line = points_lines[ind]
            geo_cur[points_line[1]['position']] = \
                self.centroid.loc[points_line[1]['index_right']]['geometry'].coords[
                    0]
        return LineString(geo_cur)

    def my_spatial_join(self, deadend_lines, deadend_pnts, line_name):
        # Spatial join between roundabout centroid to nearby dead end lines

        s_join = gpd.sjoin_nearest(left_df=deadend_pnts, right_df=self.centroid, how='left', max_distance=100,
                                   distance_col='dist').dropna(subset='dist')

        # Deadened lines from both lines should be removed
        lines_to_delete_test = s_join['line_name'].unique()  # all the Deadened lines close to roundabout

        # All deadened lines from both lines
        deads_both_side = self.dead_end_fd['line_name'].value_counts()
        deads_both_side = deads_both_side[deads_both_side == 2]

        # Remove this lines from the database
        lines_to_delete = deads_both_side[deads_both_side.index.isin(lines_to_delete_test)]

        self.network = self.network[
            ~((self.network[line_name].isin(lines_to_delete.index)) & (self.network.length < 300))]
        deadend_lines = deadend_lines[
            ~((deadend_lines[line_name].isin(lines_to_delete.index)) & (deadend_lines.length < 300))]
        # Update the geometry so the roundabout will be part of the line geometry
        change_geo = deadend_lines.copy()

        change_geo['geometry'] = change_geo.apply(lambda x: self.__update_geometry(x, s_join), axis=1)

        return change_geo
