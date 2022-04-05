new_id <- function(colum){
  org_id <- unique(colum)
  copy_col <- colum
  for(i in 1:length(org_id)){
    colum[which(copy_col == org_id[i])] = i
  }
  return(colum)
}

rad <- function(angle) 2*pi*angle/360

processing_data <- function(df){
  df <- df %>%
    group_by(hittype) %>%
    mutate(vangle_A = ifelse(is.na(vangle_A),
                             mean(vangle_A, na.rm=T),
                             vangle_A))
  df <- df %>%
    group_by(hittype) %>%
    mutate(vangle_B = ifelse(is.na(vangle_B),
                             mean(vangle_B, na.rm=T),
                             vangle_B))
  df$speed <- (df$speed_A + df$speed_B)/2
  df$vangle <- (df$vangle_A + df$vangle_B)/2
  
  df$batter <- new_id(df$batter)
  df$pitcher <- new_id(df$pitcher)
  
  return(df)
}