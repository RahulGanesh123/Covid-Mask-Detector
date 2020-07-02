import os, cv2, requests
from requests import exceptions

folder_name = 'C:\Users\rahul\Projects\Covid-Mask-Detector\Scripts\Masks' # you will need to change this
#path_name = '/content/gdrive/My Drive/data/masks'

if not os.path.exists(os.path.dirname(folder_name)):
    try:
        os.makedirs(os.path.dirname(folder_name))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#Defining/Initializing Variables
EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])

API_KEY = '7873e4c79d29431dbac945559fe5d5f3' #change API key
MAX_RESULTS = 250
GROUP_SIZE = 50
 
# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}

#Search String
search_term = 'people in stores'

#program to read api  + return result
params = {"q": search_term, "offset": 0, "count": GROUP_SIZE}
search = requests.get(URL, headers=headers, params=params)
results = search.json()

print("{}".format(results))

estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)

print("[INFO] {} total results for '{}'".format(estNumResults, search_term))
total = 0

for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
    
    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
#             print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)

            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([path_name, "{}{}".format(
                str(total).zfill(8), ext)])
            print(p)  
            
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # catch any errors that would not unable us to download the image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue
        
        # try to load the image from disk
        image = cv2.imread(p)
        if image is None:
          print("[INFO] deleting: {}".format(p))
          os.remove(p)
          continue

        total += 1
