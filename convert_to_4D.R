library(fslr)

in_base_dir = '/datasets/brats_2018'
out_base_dir = '~/Research/unet_brats/brats_2018_4D'
channel_names = c('flair', 't1', 't1ce', 't2', 'seg')

base_len = length(strsplit(in_base_dir, '/')[[1]])

sub_dirs = list.dirs(in_base_dir)
sub_dirs = grep('Brats18', sub_dirs, value=TRUE)

pb = txtProgressBar(max=length(sub_dirs), style=3)
isub = 0
for (sub_dir in sub_dirs) {
  isub = isub + 1

  subID = basename(sub_dir)
  infiles = file.path(sub_dir, paste0(subID, '_', channel_names, '.nii.gz'))
  infiles = infiles[file.exists(infiles)]

  sub_split = strsplit(sub_dir, '/')[[1]]
  sub_rel_path = do.call(file.path, as.list(sub_split[(base_len + 1) : length(sub_split)]))
  out_path = file.path(out_base_dir, sub_rel_path, paste0(subID, '_4D'))
  dir.create(dirname(out_path), showWarnings = FALSE, recursive=TRUE)

  fslmerge(infiles, direction='t', outfile=out_path, verbose=FALSE)
  setTxtProgressBar(pb, isub)
}
