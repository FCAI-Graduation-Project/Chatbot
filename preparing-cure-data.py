import json
# hello
data = [
    [
     {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for pepper plant with disease name bacterial spot?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for potato plant with disease name early blight?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for potato plant with disease name late blight?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name bacterial spot?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name early blight?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name late blight?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name leaf mold?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name septoria leaf spot?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name spider mites two spotted spider mites?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name target spot?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name yellowleaf curl  virus ?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    {
        "messages": [
        {
            "role": "system",
            "content": "You are a plant disease assistant, the user will give you plant name and plant disease. Give him the cure for the plant name and plant disease"
        },
        {
            "role": "user",
            "content": "What is the cure for tomato plant with disease name mosaic virus?"
        },
        {
            "role": "assistant",
            "content": """"""
        }
        ]
    },
    ]
]

json_string = json.dumps(data, indent=4)

# Specify the file path where you want to save the JSON file
file_path = "output.json"

# Write the JSON string to a file
with open(file_path, "w") as json_file:
    json_file.write(json_string)


# load the python file and use it
with open(file_path, "r") as json_file:
    # Load the JSON data from the file
    loaded_data = json.load(json_file)

# Now you can use the loaded_data as a Python dictionary
print("Loaded JSON data:")
print(loaded_data)