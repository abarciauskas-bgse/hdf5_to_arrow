import concurrent.futures
import h5coro
from h5coro import s3driver, filedriver
import pyarrow as pa
import numpy as np
from pydantic import BaseModel
from typing import Dict, Optional, Union

class HDF5Field(BaseModel):
    dataset: str
    hyperslice: list[tuple] = []

class HDF5Datasets(BaseModel):
    latitude: HDF5Field
    longitude: HDF5Field
    variables: Dict[str, HDF5Field]
    
class AWSCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str

class HDF5ArrowTableCreator:
    def __init__(self, uri: str, datasets: Union[HDF5Datasets, dict], credentials: Union[AWSCredentials, dict] = None, mask_using: str = None):
        """
        Initializes the HDF5ArrowTableCreator with the necessary configurations.
        
        :param uri: str - S3 URL
        :param datasets: HDF5Datasets - dictionary of HDF5 datasets to read from the HDF5 file using h5coro. Must include latitude and longitude fields.
        :param credentials: AWSCredentials - AWS credentials for accessing the S3 bucket
        :param mask_using: str - if you want to mask the table, say with a fill_value
        """
        self.uri = uri
        if isinstance(datasets, dict):
            datasets = HDF5Datasets(**datasets)
        self.datasets = datasets
        if isinstance(credentials, dict):
            credentials = AWSCredentials(**credentials)
        self.credentials = credentials
        self.mask_using = mask_using

        self.h5obj = self._initialize_h5coro()
        self.promise = self._read_datasets()

    # TODO: Add http driver option
    def _initialize_h5coro(self) -> h5coro.H5Coro:
        """Initialize the h5coro object with the provided uri and credentials."""
        if self.uri.startswith('s3://') and not self.credentials:
            raise ValueError("AWS credentials are required for S3 URLs.")
        
        args = dict(
            errorChecking=True,
            verbose=False
        )
        if self.uri.startswith('s3://') and self.credentials:
            args['resource'] = self.uri.replace('s3://', '')
            args['driverClass'] = s3driver.S3Driver
            args['credentials'] = dict(self.credentials)
        else:
            args['resource'] = self.uri
            args['driverClass'] = filedriver.FileDriver
        return h5coro.H5Coro(**args)

    def _read_datasets(self) -> dict:
        """Read datasets from the HDF5 file using h5coro."""
        variable_paths = self.datasets.variables
        dataset_objs = [dict(v) for k, v in variable_paths.items()] + [dict(self.datasets.longitude), dict(self.datasets.latitude)]
        return self.h5obj.readDatasets(dataset_objs, block=False)

    def _create_mask(self) -> np.ndarray:
        """Create a mask for the datasets if masking is requested."""
        if not self.mask_using:
            return None
        
        mask_dataset_path = self.datasets.variables[self.mask_using].dataset
        mask_dataset_points = self.promise[mask_dataset_path]
        attributes = self.h5obj.list(mask_dataset_path)[1]
        fill_value = attributes['_FillValue']
        mask = (mask_dataset_points != fill_value)
        
        return mask if mask.any() else None

    def _process_datasets(self, mask: np.ndarray):
        """Process each dataset, apply mask if necessary, and create pyarrow arrays and schema fields."""
        schema_fields = []
        pyarrow_arrays = {}

        for k, v in self.datasets.variables.items():
            if k == self.mask_using:
                continue
            
            variable_points = self.promise[v.dataset]
            if mask is not None:
                variable_points = variable_points[mask]
            
            schema_fields.append(pa.field(k, pa.from_numpy_dtype(variable_points.dtype)))
            pyarrow_arrays[k] = pa.array(variable_points)

        if self.mask_using:
            mask_dataset_path = self.datasets.variables[self.mask_using].dataset
            mask_dataset_points = self.promise[mask_dataset_path][mask]
            schema_fields.append(pa.field(self.mask_using, pa.from_numpy_dtype(mask_dataset_points.dtype)))
            pyarrow_arrays[self.mask_using] = pa.array(mask_dataset_points)

        return pyarrow_arrays, schema_fields

    # Question: Does it make sense to add other geometry options, like polygon, line, etc.
    def _create_geometry_column(self, mask: np.ndarray) -> pa.FixedSizeListArray:
        """Create a geometry column from latitude and longitude."""
        latitude = self.promise[self.datasets.latitude.dataset]
        longitude = self.promise[self.datasets.longitude.dataset]
        
        if mask is not None:
            latitude = latitude[mask]
            longitude = longitude[mask]

        np_coords = np.column_stack([longitude, latitude])
        flat_coords = pa.array(np_coords.flatten("C"))
        return pa.FixedSizeListArray.from_arrays(flat_coords, 2)

    def _build_table(self, pyarrow_arrays: dict, schema_fields: list, geom_points: pa.FixedSizeListArray) -> pa.Table:
        """Construct the final pyarrow Table with the geometry column."""
        geom_field = pa.field("geometry", geom_points.type, metadata={b'ARROW:extension:name': b'geoarrow.point'})
        schema_fields.append(geom_field)
        
        schema = pa.schema(schema_fields)
        return pa.Table.from_arrays([*pyarrow_arrays.values(), geom_points], schema=schema)

    def create_table(self) -> pa.Table:
        """
        Public method to create the pyarrow table from the HDF5 file.
        
        :return: pa.Table - pyarrow Table with a geometry column of latitude, longitude points
        """
        mask = self._create_mask()
        pyarrow_arrays, schema_fields = self._process_datasets(mask)
        geom_points = self._create_geometry_column(mask)
        
        return self._build_table(pyarrow_arrays, schema_fields, geom_points)


def create_table(
    uri: str,
    datasets: HDF5Datasets,
    credentials: AWSCredentials,
    mask_using: str = None
) -> pa.Table:
    creator = HDF5ArrowTableCreator(uri, datasets, credentials, mask_using)
    return creator.create_table()
    
def concat_tables(uris: list[str], datasets: HDF5Datasets, credentials: AWSCredentials, mask_using: str = None) -> pa.Table:
    """
    Concatenate multiple pyarrow tables into a single table.
    Must have the same schema.
    """
    tables = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        shared_args = dict(datasets=datasets, mask_using=mask_using, credentials=credentials)
        futures = [executor.submit(create_table, uri=uri, **shared_args) for uri in uris]
        completed_futures, _ = concurrent.futures.wait(futures) 
        for future in completed_futures:
            try:
                table = future.result()
                if table:
                    tables.append(table)
            except Exception as exception:
                print(exception) 
            
        return pa.concat_tables(tables)      

# Example usage:
# Generate credentials
# Query for data uris (S3 URLs)
# Generate datasets
# Create 1 table from 1 file:
# > creator = HDF5ArrowTableCreator(uri, datasets, credentials, mask_using)
# > table = creator.create_table()
# Create 1 table from multiple files:
# > table = concat_tables(uris, datasets, credentials, mask_using)
#
