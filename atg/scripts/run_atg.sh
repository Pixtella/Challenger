python3 run_es.py --batch_in scenes_in/batch_0010.pkl \
                --batch_out scenes_out/deb/batch_0010_out.pkl \
                --ckpt ckpts/checkpoints/ig360_adv.ckpt \
                --obj 1 \
                --temperature 1.0 \
                --cem_iters 5 \
                --seed 0