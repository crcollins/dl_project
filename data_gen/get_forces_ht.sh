for i in {000001..050000}; do echo $i; grep "Forces (Har" benzene_md2/benzene_$i.log -A14 | tail -n12 | cut -d ' ' -f 24- | tr '\n' ' ' | sed -e 's/^/\n/' >> benzene_forces_ht.txt ; done
