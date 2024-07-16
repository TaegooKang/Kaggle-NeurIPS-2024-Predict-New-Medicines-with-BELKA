
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
                    main_train.py -ex Leash-BELKA-3d-graph