# Exploration of arxiv datasets

## Data
The data are collected by *arxiv_collector*.
Place the downloaded files in the data folder
so that they can be found by the training script. 

## Exploration and training 
For exploration of the data, you can use the 
*test_embedding_2.ipynb* script which is updated
to use Tensorflow 1.15. After this step you can
also train the final model using *train_data.py*

## Deployment

We use AWS to host this project. For large files, we use the Amazon S3.

To upload the files, we first create a `s3 bucket`:
```
aws s3 mb s3://ml-bucket-1
```
Here `ml-bucket-1` is the bucket name. We should also include this name in our `zappa_settings.json` such that our script can get back our files on s3.

###  Virtual environment

```
./zappa.commands.sh
```

The `webapp` is quite large, we can reduce its size a bit. 

```
find . -name "tests" | xargs rm -rf
find . -name "*.pyc" -delete
find . -name "*.so" | xargs strip
```

We use `zappa` to deploy the package on the `Amazon Lambda`. We first run `zappa init` to initialize the project. Here are our sample configuration file:

```json
{
    "dev": {
        "app_function": "webapp.app",
        "aws_region": "us-east-1",
        "profile_name": "default",
        "project_name": "webapp",
        "runtime": "python3.6",
        "s3_bucket": "YOUR-ZAPPA-BUCKET-NAME",
	    "slim_handler": true,
	    "use_precompiled_packages": false,
        "memory_size": 1024
    }
}
```

Note that we have added the last three options: `slim_handler`, `use_precompiled_packages`, and `memory_size`.




