for i in {000001..050000}; do cat header benzene_md/benzene_$i.xyz footer | sed -e "s/name/benzene_$i/" | sed -e '8,9d' > benzene_md2/benzene_$i.gjf; done
