      list=`gcloud container images list-tags gcr.io/iwasnothing-self-learning/osrsi --filter="NOT targs:latest" --format=yaml|grep digest|sed -e "s/^digest: //g"`
      for md in $list ; do 
            gcloud container images delete gcr.io/iwasnothing-self-learning/osrsi@$md --force-delete-tags --quiet
      done