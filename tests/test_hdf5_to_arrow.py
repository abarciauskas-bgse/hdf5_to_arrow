#!/usr/bin/env python

"""Tests for `hdf5_to_arrow` package."""
import os
import h5py
import pyarrow as pa
import numpy as np
from tempfile import NamedTemporaryFile
from unittest import TestCase
from hdf5_to_arrow import hdf5_to_arrow

class TestHDF5ArrowTableCreator(TestCase):

    def setUp(self):
        # Create a temporary HDF5 file for testing
        self.temp_file = NamedTemporaryFile(delete=False)
        self.hdf5_file = h5py.File(self.temp_file.name, 'w')

        # Create datasets for latitude, longitude, and other variables
        lat_data = np.array([34.05, 36.16, 40.71], dtype=np.float32)
        lon_data = np.array([-118.24, -115.15, -74.00], dtype=np.float32)
        temp_data = np.array([20.5, 21.0, 19.8], dtype=np.float32)
        
        self.hdf5_file.create_dataset('latitude', data=lat_data)
        self.hdf5_file.create_dataset('longitude', data=lon_data)
        self.hdf5_file.create_dataset('temperature', data=temp_data)
        
        # Add fill_value attribute to the temperature dataset
        self.hdf5_file['temperature'].attrs['_FillValue'] = np.float32(-9999.0)
        
        self.hdf5_file.close()

        # Prepare the datasets dictionary for HDF5ArrowTableCreator
        self.datasets_dict = {
            "variables": {
                "temperature": {"dataset": "temperature"}
            },
            "longitude": {"dataset": "longitude"},
            "latitude": {"dataset": "latitude"}
        }

    def tearDown(self):
        # Clean up the temporary file
        os.remove(self.temp_file.name)

    def test_create_table(self):
        # Create an instance of the HDF5ArrowTableCreator class
        creator = hdf5_to_arrow.HDF5ArrowTableCreator(
            uri=self.temp_file.name,  # Local file URI
            datasets=self.datasets_dict,
            mask_using=None  # No masking in this test
        )

        # Create the table
        table = creator.create_table()

        # Verify the table structure and content
        self.assertIsInstance(table, pa.Table)
        self.assertEqual(table.num_columns, 2)  # temperature, geometry
        self.assertEqual(table.num_rows, 3)     # Three rows of data

        # Check if the geometry column is correctly created
        geom_column = table['geometry']
        self.assertEqual(len(geom_column), 3)
        self.assertTrue(isinstance(geom_column, pa.lib.ChunkedArray))

        # Verify the data in the temperature column
        temp_column = table['temperature']
        self.assertTrue(np.array_equal(temp_column.to_numpy(), np.array([20.5, 21.0, 19.8], dtype=np.float32)))

        # Verify the data in the longitude and latitude columns
        geom_column_numpy = table['geometry'].to_numpy()
        self.assertTrue(np.array_equal(geom_column_numpy[0], np.array([-118.24, 34.05], dtype=np.float32)))
        self.assertTrue(np.array_equal(geom_column_numpy[1], np.array([-115.15, 36.16], dtype=np.float32)))
        self.assertTrue(np.array_equal(geom_column_numpy[2], np.array([-74.0, 40.71], dtype=np.float32)))


if __name__ == '__main__':
    import unittest
    unittest.main()
