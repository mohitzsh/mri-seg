from subprocess import call

# ############
# #Process 2_T1.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/1_T1.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/1_T1_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # register
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/1_T1_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/1_T1_iso_reg.nii",
#     "--out_affine","data/3d_MRBrains/img/temp/1_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "reg",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/1_T1_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/111_T1.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/1_T1_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/11_T1.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process 2_T1_IR.nii #
# ############
# iso_img
call(
[
    "python", "scripts/process_vols.py",
    "--in_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/4_T1_IR.nii",
    "--out_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso.nii",
    "--mode", "iso_img",
]
)
# resample_img
call(
[
    "python", "scripts/process_vols.py",
    "--in_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso.nii",
    "--out_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg.nii",
    "--in_affine","/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_affine.txt",
    "--atlas","/scratch/halstead/s/sharm267/mri/mni_icbm152_t1_tal_nlin_sym_09c.nii",
    "--mode", "resample_img",
]
)
# sgauss
call(
[
    "python", "scripts/process_vols.py",
    "--in_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg.nii",
    "--out_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg_sgauss.nii",
    "--mode", "sgauss",
]
)
# ahe
call(
[
    "python", "scripts/process_vols.py",
    "--in_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg_sgauss.nii",
    "--out_name", "/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg_sgauss_ahe.nii",
    "--mode", "ahe",
]
)
call(
[
    "python","scripts/process_vols.py",
    "--in_name","/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/temp/4_T1_IR_iso_reg_sgauss_ahe.nii",
    "--out_name","/scratch/halstead/s/sharm267/mri/3d_MRBrains/img/44_T1_IR_shifted.nii",
    "--mode","center"
]
)
# ############
# #Process 2_T2_FLAIR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/1_T2_FLAIR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/1_T2_FLAIR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/1_T2_FLAIR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/1_T2_FLAIR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/1_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/1_T2_FLAIR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/111_T2_FLAIR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/1_T2_FLAIR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/11_T2_FLAIR.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process cls/2_T1.nii #
# ############
# # iso_img
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/1.nii",
# #     "--out_name", "data/3d_MRBrains/cls/temp/1.nii",
# #     "--mode", "iso_cls",
# # ]
# # )
# # # resample_
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/temp/1.nii",
# #     "--out_name", "data/3d_MRBrains/cls/11.nii",
# #     "--in_affine","data/3d_MRBrains/img/temp/1_affine.txt",
# #     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
# #     "--mode", "resample_cls",
# # ]
# # )
#
#
#
#
#
#
# ############
# #Process 2_T1.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/2_T1.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T1_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # register
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T1_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T1_iso_reg.nii",
#     "--out_affine","data/3d_MRBrains/img/temp/2_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "reg",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T1_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/222_T1.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/2_T1_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/22_T1.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process 2_T1_IR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/2_T1_IR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T1_IR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T1_IR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T1_IR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/2_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T1_IR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/222_T1_IR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/2_T1_IR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/22_T1_IR.nii",
# #     "--mode", "ahe",
# # ]
# # )
# ############
# #Process 2_T2_FLAIR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/2_T2_FLAIR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T2_FLAIR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T2_FLAIR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/2_T2_FLAIR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/2_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/2_T2_FLAIR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/222_T2_FLAIR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/2_T2_FLAIR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/22_T2_FLAIR.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process cls/2_T1.nii #
# ############
# # iso_img
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/2.nii",
# #     "--out_name", "data/3d_MRBrains/cls/temp/2.nii",
# #     "--mode", "iso_cls",
# # ]
# # )
# # # resample_
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/temp/2.nii",
# #     "--out_name", "data/3d_MRBrains/cls/22.nii",
# #     "--in_affine","data/3d_MRBrains/img/temp/2_affine.txt",
# #     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
# #     "--mode", "resample_cls",
# # ]
# # )
#
#
# ############
# #Process 2_T1.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/3_T1.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T1_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # register
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T1_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T1_iso_reg.nii",
#     "--out_affine","data/3d_MRBrains/img/temp/3_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "reg",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T1_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/333_T1.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/3_T1_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/33_T1.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process 2_T1_IR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/3_T1_IR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T1_IR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T1_IR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T1_IR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/3_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T1_IR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/333_T1_IR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/3_T1_IR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/33_T1_IR.nii",
# #     "--mode", "ahe",
# # ]
# # )
# ############
# #Process 2_T2_FLAIR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/3_T2_FLAIR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T2_FLAIR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T2_FLAIR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/3_T2_FLAIR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/3_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/3_T2_FLAIR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/333_T2_FLAIR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/3_T2_FLAIR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/33_T2_FLAIR.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process cls/2_T1.nii #
# ############
# # iso_img
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/3.nii",
# #     "--out_name", "data/3d_MRBrains/cls/temp/3.nii",
# #     "--mode", "iso_cls",
# # ]
# # )
# # # resample_
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/cls/temp/3.nii",
# #     "--out_name", "data/3d_MRBrains/cls/33.nii",
# #     "--in_affine","data/3d_MRBrains/img/temp/3_affine.txt",
# #     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
# #     "--mode", "resample_cls",
# # ]
# # )
# #
#
# ############
# #Process 4_T1.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/4_T1.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T1_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # register
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T1_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T1_iso_reg.nii",
#     "--out_affine","data/3d_MRBrains/img/temp/4_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "reg",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T1_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/444_T1.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/4_T1_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/44_T1.nii",
# #     "--mode", "ahe",
# # ]
# # )
#
# ############
# #Process 2_T1_IR.nii #
# ############
# # iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/4_T1_IR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T1_IR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T1_IR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T1_IR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/4_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T1_IR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/444_T1_IR.nii",
#     "--mode", "sgauss",
# ]
# )
# # # ahe
# # call(
# # [
# #     "python", "scripts/process_vols.py",
# #     "--in_name", "data/3d_MRBrains/img/temp/4_T1_IR_iso_reg_sgauss.nii",
# #     "--out_name", "data/3d_MRBrains/img/44_T1_IR.nii",
# #     "--mode", "ahe",
# # ]
# # )
# ############
# #Process 2_T2_FLAIR.nii #
# ############
# iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/4_T2_FLAIR.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso.nii",
#     "--mode", "iso_img",
# ]
# )
# # resample_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso_reg.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/4_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_img",
# ]
# )
# # sgauss
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso_reg.nii",
#     "--out_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso_reg_sgauss.nii",
#     "--mode", "sgauss",
# ]
# )
# # ahe
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/img/temp/4_T2_FLAIR_iso_reg_sgauss.nii",
#     "--out_name", "data/3d_MRBrains/img/44_T2_FLAIR.nii",
#     "--mode", "ahe",
# ]
# )

############
#Process cls/2_T1.nii #
############
# iso_img
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/cls/4.nii",
#     "--out_name", "data/3d_MRBrains/cls/temp/4.nii",
#     "--mode", "iso_cls",
# ]
# )
# call(
# [
#     "python", "scripts/process_vols.py",
#     "--in_name", "data/3d_MRBrains/cls/temp/4.nii",
#     "--out_name", "data/3d_MRBrains/cls/44.nii",
#     "--in_affine","data/3d_MRBrains/img/temp/4_affine.txt",
#     "--atlas","data/mni_icbm152_t1_tal_nlin_sym_09c.nii",
#     "--mode", "resample_cls",
# ]
# )

# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/11_T1.nii",
#     "--out_name","data/3d_MRBrains/img/11_T1_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/11_T1_IR.nii",
#     "--out_name","data/3d_MRBrains/img/11_T1_IR_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/11_T2_FLAIR.nii",
#     "--out_name","data/3d_MRBrains/img/11_T2_FLAIR_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/22_T1.nii",
#     "--out_name","data/3d_MRBrains/img/22_T1_shifted.nii",
#     "--mode","center"
# ]
# )
#
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/22_T1_IR.nii",
#     "--out_name","data/3d_MRBrains/img/22_T1_IR_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/22_T2_FLAIR.nii",
#     "--out_name","data/3d_MRBrains/img/22_T2_FLAIR_shifted.nii",
#     "--mode","center"
# ]
# )
#
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/33_T1.nii",
#     "--out_name","data/3d_MRBrains/img/33_T1_shifted.nii",
#     "--mode","center"
# ]
# )
#
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/33_T1_IR.nii",
#     "--out_name","data/3d_MRBrains/img/33_T1_IR_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/33_T2_FLAIR.nii",
#     "--out_name","data/3d_MRBrains/img/33_T2_FLAIR_shifted.nii",
#     "--mode","center"
# ]
# )
#
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/44_T1.nii",
#     "--out_name","data/3d_MRBrains/img/44_T1_shifted.nii",
#     "--mode","center"
# ]
# )
#
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/44_T1_IR.nii",
#     "--out_name","data/3d_MRBrains/img/44_T1_IR_shifted.nii",
#     "--mode","center"
# ]
# )
# call(
# [
#     "python","scripts/process_vols.py",
#     "--in_name","data/3d_MRBrains/img/44_T2_FLAIR.nii",
#     "--out_name","data/3d_MRBrains/img/44_T2_FLAIR_shifted.nii",
#     "--mode","center"
# ]
# )
