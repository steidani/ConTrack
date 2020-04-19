program contrack

! ----------------------------------------------------------------------
!
! Date: 1st January 2011
!                                                              
!
! ------->> Purpose: Spatial and temporal contour tracking programm
!             
! This programm is a completely re-written new version of the tracking 
! programm used for the studies in e.g. Schwierz et al. (2005), 
! Croci-Maspoli et al. (2007a,b), Croci-Maspoli et al. (2009).
!
! ------->> What is necessary to compile contrack?

! - contrack has been compiled with gfortran 
!   (included in the GCC family, http://gcc.gnu.org/ ). Feel free
!   to test other fortran compilers.
! - contrack requires the netcdf-4.0.1 library (also compiled with gfortran).
! - contrack comes with a simple make-file (you have to adjust this file
!   before compiling).
!
! ------->> What netCDF-file is necessary as input?
!
! - a single CF-netCDF-file (http://cf-pcmdi.llnl.gov/) including all required
!   time steps.
! - a longitude, latitude and time dimension.
! - contrack does not calculate a climatological anomaly field. This needs to
!   be calculated before running contrack.
!
! ------->> How does the output file look like?
!
! - the output CF-netCDF file has the same dimensions as the input netCDF file 
! - the output has a binary format (0 and 1), whereas 1 captures regions of
!   the blocks.
!   
! ------->> Known issues or features not implemented yet in contrack 0.8:
!
! 1) The calculation and saving of common statistical blocking features 
!    (e.g. blocking size, blocking length, blocking path, ...) is not 
!    implemented yet. My intention is to save these information directly 
!    on the netCDF file.
! 2) Only two-dimensional netCDF-files can be used as input file (longitude,
!    latitude and time). The code can not handle a third dimension (e.g.
!    vertical) on the same netCDF-file yet.
! 3) The asymmetry of the blocking features is not implemented yet.
! 3) ... more to come!
!
! ------->>  Version control

! Version 0.1-0.8: May 2010 - fortran 90 compatible
!                           - CF-netCDF compatible (instead of IVE-netCDF)
!                           - new tracking algorithm
!                  Oct 2011 - started with handling also different vertical
!                             levels but not finished yet.
!                  Jan 2011 - added these explanations
!
! If you have any comments please send them to mischa.croci-maspoli@alumni.ethz.ch.
!
! This programm code comes with no guarantee and is currently still 
! under construction. Be aware that no extensive testing and comparison to the
! old code has been performed yet. 
!
! Copyright: mischa.croci-maspoli@alumni.ethz.ch
!
! Modifications SP: omit first and last n time steps in output,
!                   where n=persistence;
!                   comment out allocation etc. related to vertical dimension
!
! ----------------------------------------------------------------------

USE netcdf 

implicit none

integer, external :: iargc

integer :: i,j,t,k,ii,jj,tt,counter
integer :: persistence, verbose, calcmeranom, cont
integer :: titleLength, commentLength, sourceLength, institLength
integer :: errmsg, outarr
integer :: nrcontours
integer :: ttbegin, ttend
real    :: fraction_backward, fraction_forward, areacon, areaover_forward, areaover_backward, overlap, threshold

character*(100) infilename,outfilename,arg
character*(100) outvarname,outstandvarname, outvarunit
character*(100) invarname, inlat, inlon, intime, inver
character*(100) contrackversion
character*(5) gorl

character*(1000) cfcomment, cftitle, cfsource, cfinstit
character*(1000) cfoutcomment, cfouttitle, cfoutsource, cfoutinstit
integer :: ncid, status, nDims, nVars, nGlobalAtts, unlimDimID
integer :: label, nrsmooth, nrg, nrgc
integer :: twosided

integer :: LatDimID,LonDimID, TimeDimID,timeVarID,lonVarID,latVarID,vorpotVarID
integer :: VerDimID
integer :: varLatID,varLonID,varPVID,varTimeID
integer :: nLats,nLong,nTimes,nVert

real :: dlon, dlat

integer, dimension(:), allocatable :: times, longi
real, dimension(:), allocatable ::  latitude, longitude
integer, dimension(:), allocatable :: vertical
integer, dimension(:), allocatable :: iiv, jjv, ttv
integer, dimension(:), allocatable :: time_array
real, dimension(:,:,:), allocatable :: inarr, arr, arr1, arr2
real, dimension(:,:), allocatable :: latmean


! --------------------------------------------------------------
! input handling
! --------------------------------------------------------------

contrackversion = 'version 0.8'
status = 0

print*, '-------------------------------------------------------'
print*, 'contrack ', trim(contrackversion)
print*, '-------------------------------------------------------'

999 continue

if ( (iargc().ne.1).or.(status.gt.0) ) then
   print*, '-------------------------------------------------------'
   print*, 'contrack ', trim(contrackversion)
   print*, '-------------------------------------------------------'
   print*, 'Usage: contrack inputfile'
   print*, '-------------------------------------------------------'
   print*, 'The inputfile needs the following structure:'
   print*, ''
   print*, 'infilename   -> name of the input CF-netCDF file'
   print*, 'invarname    -> name of the netCDF variable name'
   print*, 'inlat        -> name of the netCDF latitude variable'
   print*, 'inlon        -> name of the netCDF longitude variable'
   print*, 'intime       -> name of the netCDF time variable'
   print*, 'threshold    -> threshold value to detect contours'
   print*, 'gorl         -> find contours that are greater or '
   print*, '                lower threshold value [ge,le,gt,lt]'
   print*, 'persistence  -> temporal persistence (in time steps)'
   print*, '                of the contour life time'
   print*, 'overlap      -> overlapping fraction of two contours'
   print*, '                between two time steps [0-1]'
   print*, 'nrsmooth     -> temporal smoothing of the infilename'
   print*, '                (in time steps)'
   print*, 'outarr       -> values of the contours in the output array'
   print*, '                0: contours are numbered by each contour'
   print*, '                1: contours are labeled 0 and 1'
   print*, 'outfilename  -> name of the output CF-netCDF file'
   print*, 'outvarname   -> name of the output netCDF variable name'
   print*, 'outstvarname -> name of the output netCDF standard'
   print*, '                variable name'
   print*, 'outvarunit   -> name of the outuput netCDF unit'
   print*, 'verbose      -> [0 or 1]'
   print*, 'calcmeranom  -> flag whether to calc meriodonal anomalies'
   print*, '                or not [0 or 1]'
   print*, 'twosided     -> twosided overlap test [0 or 1]'
   print*, '-------------------------------------------------------'
   print*, 'Please send any comments / bugs to:'
   print*, 'mischa.croci-maspoli@alumni.ethz.ch'
   print*, '-------------------------------------------------------'
   print*, ''
   print*, '-->> contrack not executed, an error occured!'
   print*, ''
   call exit(1)
endif

! Read inputfile
! --------------
call getarg(1,arg)
infilename=trim(arg)


! Open inputfile and read variables
! ---------------------------------
open(2,file=infilename,status='old',iostat=status)
 if ( status.gt.0 ) then
    print*, ''
    print*, '-> Problem in contrack: Your ASCII input file "', trim(infilename), &
            '" can not be found'
    print*, '-> For further information use ./contrack only'
    print*, ''
    stop
 endif

 read(2,*,iostat=status) infilename
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) invarname
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) inlat
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) inlon
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) intime
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) threshold
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) gorl
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) persistence
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) overlap
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) nrsmooth
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) outarr
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) outfilename
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) outvarname
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) outstandvarname
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) outvarunit 
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) verbose
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) calcmeranom
 if ( status.gt.0 ) goto 999
 read(2,*,iostat=status) twosided
 if ( status.gt.0 ) goto 999
close(2)


! --------------------------------------------------------------
! read CF-netCDF file
! --------------------------------------------------------------

allocate(time_array(8))
call date_and_time (values=time_array)
write(*,100), "-> Read ", trim(infilename), time_array(5), ':', time_array(6), ':', time_array(7)
100 format(A,A,I5,A,I2,A,I2)

! Initially set error to indicate no errors.
status = 0

! open netCDF file
status = nf90_open(infilename, nf90_nowrite, ncid)

! get dimensions
status = nf90_inq_dimid(ncid, inlat, LatDimID)
status = nf90_inquire_dimension(ncid, LatDimID, len = nLats)

status = nf90_inq_dimid(ncid, inlon, LonDimID)
status = nf90_inquire_dimension(ncid, LonDimID, len = nLong)

!status = nf90_inq_dimid(ncid, inver, VerDimID)
!status = nf90_inquire_dimension(ncid, VerDimID, len = nVert)

status = nf90_inq_dimid(ncid, intime, TimeDimID)
status = nf90_inquire_dimension(ncid, TimeDimID, len = nTimes)

! allocate arrays
allocate(times(nTimes))
allocate(latitude(nLats))
allocate(longitude(nLong))
!allocate(vertical(nVert))
allocate(inarr(nLong,nLats,nTimes))
allocate(arr(nLong,nLats,nTimes))
allocate(arr1(nLong,nLats,nTimes))
allocate(arr2(nLong,nLats,nTimes))
allocate(latmean(nLats,nTimes))
allocate(iiv(nLats*nLong*nTimes))
allocate(jjv(nLats*nLong*nTimes))
allocate(ttv(nLats*nLong*nTimes))


! get variables
status = nf90_inq_varid(ncid, intime, timeVarID)
status = nf90_get_var(ncid, timeVarID, times)

status = nf90_inq_varid(ncid, inlon, lonVarID)
status = nf90_get_var(ncid, lonVarID, longitude)

status = nf90_inq_varid(ncid, inlat, latVarID)
status = nf90_get_var(ncid, latVarID, latitude)

status = nf90_inq_varid(ncid, invarname, vorpotVarID)
status = nf90_get_var(ncid, vorpotVarID, inarr)

status = nf90_inquire_attribute(ncid, nf90_global, "comment", len = commentLength)
status = nf90_get_att(ncid, NF90_GLOBAL, 'comment', cfcomment)
status = nf90_inquire_attribute(ncid, nf90_global, "title", len = titleLength)
status = nf90_get_att(ncid, NF90_GLOBAL, 'title', cftitle)
status = nf90_inquire_attribute(ncid, nf90_global, "source", len = sourceLength)
status = nf90_get_att(ncid, NF90_GLOBAL, 'source', cfsource)
status = nf90_inquire_attribute(ncid, nf90_global, "institution", len = institLength)
status = nf90_get_att(ncid, NF90_GLOBAL, 'institution', cfinstit)

status = nf90_close(ncid)

! interpolate missing values at the pole
do t=1,nTimes

    inarr(:,1,t) = sum(inarr(:,2,t))/nLong
    inarr(:,(nLats-1):nLats,t) = sum(inarr(:,nLats-2,t))/nLong  

end do


! compute dlon dlat
dlon = 360.0 / (nLong - 1)
dlat = 180.0 / (nLats - 1)

!print*,latitude

! --------------------------------------------------------------
! Calculate meridional anomaly
! --------------------------------------------------------------

if ( calcmeranom.eq.1 ) then

print*, '-->> calculate meridional mean anomaly'

! Calculate latitudinal mean
! --------------------------------------------------------------
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong
   latmean(j,t) = latmean(j,t) + inarr(i,j,t)
enddo
enddo
enddo

! Subtract latitudinal mean
! --------------------------------------------------------------
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong
  arr(i,j,t) = inarr(i,j,t) - latmean(j,t)/nLong
enddo
enddo
enddo


! Calculate running means (centered at the first file)
! --------------------------------------------------------------
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong
  do k = 0,nrsmooth
     arr1(i,j,t) = arr1(i,j,t) + arr(i,j,t+k)
  enddo
enddo
enddo
enddo

arr = arr1/nrsmooth

else
  print*, '-->> no anomaly calculated'
  arr(:,:,:) = inarr(:,:,:)
endif


! Define closed contours by a threshold value
! --------------------------------------------------------------

print*, '-->> define contours with threshold value'

!$omp parallel do
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

  if ( trim(gorl).eq.'ge' ) then
     if ( arr(i,j,t).ge.threshold) then
        arr(i,j,t) = 1
     else
        arr(i,j,t) = 0
     endif
  endif
  if ( trim(gorl).eq.'le' ) then
     if ( arr(i,j,t).le.threshold) then
        arr(i,j,t) = 1
     else
        arr(i,j,t) = 0
     endif
  endif
  if ( trim(gorl).eq.'gt' ) then
     if ( arr(i,j,t).gt.threshold) then
        arr(i,j,t) = 1
     else
        arr(i,j,t) = 0
     endif
  endif
  if ( trim(gorl).eq.'lt' ) then
     if ( arr(i,j,t).lt.threshold) then
        arr(i,j,t) = 1
     else
        arr(i,j,t) = 0
     endif
  endif

enddo
enddo
enddo
!$omp end parallel do


! Identify individual contour areas on a 2D-space (only x-y, no time)
! -------------------------------------------------------------------

print*, '-->> identify individual contours'

arr2(:,:,:) = 0    ! set arr2 to zero

do t = 1,nTimes
label = 1          ! begin labeling with 1
do j = 1,nLats
do i = 1,nLong

  if ( (arr(i,j,t).eq.1).and.(arr2(i,j,t).lt.1) ) then

     arr2(i,j,t) = label    ! first identified grid point in countour
     nrg         = 1        ! number of grid points
     nrgc        = 1        ! number of grid points counter
     iiv(nrgc)   = i        ! x-dir coordinates within contour
     jjv(nrgc)   = j        ! y-dir coordinates within contour

     do while ( nrgc.le.nrg )

       do ii=-1,1
       do jj=-1,1

        if (  (arr(iiv(nrgc)+ii,jjv(nrgc)+jj,t).eq.1).and. &
             (arr2(iiv(nrgc)+ii,jjv(nrgc)+jj,t).ne.label) ) then

             arr2(iiv(nrgc)+ii,jjv(nrgc)+jj,t) = label
             nrg                               = nrg + 1
             iiv(nrg)                          = iiv(nrgc)+ii
             jjv(nrg)                          = jjv(nrgc)+jj

        endif

       enddo
       enddo

     nrgc = nrgc + 1

     enddo


  label = label + 1

  endif

enddo
enddo
enddo


! Define temporal overlapping
! --------------------------------------------------------------

print*, '-->> overlapping'

!$omp parallel do private(areacon, areaover_forward, areaover_backward, fraction_backward, fraction_forward)
do t  = 2,nTimes-1
do ii = 1,int(maxval(arr2(:,:,t)))   ! loop over the number of individual contours

areacon  = 0.
areaover_forward  = 0.
areaover_backward = 0.
fraction_backward = 0.
fraction_forward = 0.

  do i  = 1,nLong
  do j  = 1,nLats

      if ( ( arr2(i,j,t).eq.ii) ) then
         areacon = areacon+(111 * dlat * (111 * dlon * cos(3.14159 / 180. * latitude(j))))               ! cos(2*3.14159/360*j)
      endif

      if ( ( arr2(i,j,t).eq.ii).and.(arr(i,j,t+1).ge.1) ) then
         areaover_forward = areaover_forward+(111 * dlat * (111 * dlon * cos(3.14159 / 180. * latitude(j))))
      endif
      
      if ( ( arr2(i,j,t).eq.ii).and.(arr(i,j,t-1).ge.1) ) then
         areaover_backward = areaover_backward+(111 * dlat * (111 * dlon * cos(3.14159 / 180. * latitude(j))))
      endif

  enddo
  enddo

  fraction_backward = 1 / areacon * areaover_backward
  fraction_forward = 1 / areacon * areaover_forward
  !print*, ii, int(maxval(arr2(:,:,t))), fraction
  
  !print*,areacon,areaover,fraction

  do i  = 1,nLong
  do j  = 1,nLats
     if (twosided .eq. 1) then
         if ( ((fraction_backward.lt.overlap).or.(fraction_forward.lt.overlap)).and.(arr2(i,j,t).eq.ii) ) then
             arr2(i,j,t) = 0.
         endif
     else
         if ( (fraction_forward.lt.overlap).and.(arr2(i,j,t).eq.ii) ) then
             arr2(i,j,t) = 0.
         endif     
     end if
  enddo
  enddo

enddo
enddo
!$omp end parallel do

! Identify individual contour areas (3D including time)
! --------------------------------------------------------------

print*, '-->> identify 3D contours'

arr(:,:,:)  = arr2(:,:,:)

!$omp parallel do 
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

   if ( arr(i,j,t).ge.1 ) then
        arr(i,j,t) = 1
   endif

enddo
enddo
enddo
!$omp end parallel do

arr2(:,:,:) = 0    ! set arr2 to zero
label = 1          ! begin labeling with 1


do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

  if ( (arr(i,j,t).eq.1).and.(arr2(i,j,t).lt.1) ) then

     arr2(i,j,t) = label    ! first identified grid point in countour
     nrg         = 1        ! number of grid points
     nrgc        = 1        ! number of grid points counter
     iiv(nrgc)   = i        ! x-dir coordinates within contour
     jjv(nrgc)   = j        ! y-dir coordinates within contour
     ttv(nrgc)   = t        ! t-dir coordinates within contour
     ttbegin     = t        ! genesis time

!     print*, i,j,t,label

     do while ( nrgc.le.nrg )

       do ii=-1,1
       do jj=-1,1
       do tt=-1,1

        if (  (arr(iiv(nrgc)+ii,jjv(nrgc)+jj,ttv(nrgc)+tt).eq.1).and. &
             (arr2(iiv(nrgc)+ii,jjv(nrgc)+jj,ttv(nrgc)+tt).ne.label) ) then

             arr2(iiv(nrgc)+ii,jjv(nrgc)+jj,ttv(nrgc)+tt) = label
             nrg                               = nrg + 1
             iiv(nrg)                          = iiv(nrgc)+ii
             jjv(nrg)                          = jjv(nrgc)+jj
             ttv(nrg)                          = ttv(nrgc)+tt

             ! get time of lysis
             if (ttv(nrg).gt.ttend) then
                ttend = ttend + 1
             endif

        endif

       enddo
       enddo
       enddo

!       print*, iiv(nrg),jjv(nrg),ttv(nrg), label
!       print*, nrg, nrgc, ttv(nrgc)
!       if (nrgc.gt.20) stop

     nrgc = nrgc + 1

     enddo

!  print*, label, ttbegin, ttend,ttend-ttbegin
  label = label + 1


  endif

enddo
enddo
enddo


if ( verbose.eq.1 ) print*, '     -> number of individual contours: ', label-1

! Temporal persistence
! --------------------------------------------------------------

print*, '-->> temporal persistence'

!$omp parallel do private(counter)
do ii = 1,label
counter = 0
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

   if ( arr2(i,j,t).eq.ii) then
      counter = counter + 1
      goto 234
   endif

enddo
enddo
 234 continue
enddo

do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

   if ( (counter.lt.persistence ).and.(arr2(i,j,t).eq.ii) )  then
      arr2(i,j,t) = 0.
   endif

enddo
enddo
enddo

enddo
!$omp end parallel do

! get contour information
! --------------------------------------------------------------

print*, '-->> get contour information'

! number of individual contours
! length of individual contours
! mean area size of individual contours
! genesis/lysis date
! genesis/lysis area at specific date

nrcontours = 0
do t = 1,nTimes
do j = 1,nLats
do i = 1,nLong

 if ( arr2(i,j,t).gt.nrcontours ) then
  nrcontours = nrcontours + 1 
 endif

enddo
enddo
enddo

print*, '     -> number of tracked contours: ', nrcontours



! change output array to a 0/1 array
! --------------------------------------------------------------

if ( outarr.eq.1 ) then
   do t = 1,nTimes
   do j = 1,nLats
   do i = 1,nLong

      if ( arr2(i,j,t).gt.0 ) then
         arr2(i,j,t) = 1.
      endif
   enddo
   enddo
   enddo
endif


arr = arr2

11 continue


! --------------------------------------------------------------
! write CF-netCDF file
! --------------------------------------------------------------
  
call date_and_time (values=time_array)
write(*,100), "-> Write ", trim(outfilename), time_array(5), ':', time_array(6), ':', time_array(7)
   
! definitions
! --------------------------------------------------------------

cfoutinstit = cfinstit
cfouttitle  = 'contrack '//trim(contrackversion)
cfoutsource = cfsource
write(cfoutcomment, 101) 'contrack is based upon the following input file attributes: ', &
                          trim(cfcomment), &
                         ' -->> Contrack specifications:: ',&
                         'contours identified that are -', trim(gorl), '- threshold value, ', &
                         'threshold value for contour:', threshold, &
                         ', overlapping fraction:', overlap, &
                         ', persistence time steps:', persistence, &
                         ', anomalies calculated by subtracting meridional mean [1] or ', &
                         'from the input file directly [0]: ', calcmeranom

101 format(A,A,A,A,A,A,A,F8.3,A,F7.3,A,I5,A,A,I3)

! start writing 
! ----------------------------------------------------------------

! Initially set error to indicate no errors.
status = 0

! create the netCDF file
status = nf90_create(trim(outfilename), NF90_CLOBBER, ncID)
! IF (ierr.NE.0) GOTO 920

! Dimensions -----------------------------------------------------
status=nf90_def_dim(ncID,'longitude',nLong,LonDimID)
status=nf90_def_dim(ncID,'latitude',nLats,LatDimID)
status=nf90_def_dim(ncID,'time',nf90_unlimited, TimeDimID)

! Variables ------------------------------------------------------
status = nf90_def_var(ncID,'longitude',NF90_FLOAT,(/ LonDimID /),varLonID)
status = nf90_put_att(ncID, varLonID, "standard_name", "longitude")
status = nf90_put_att(ncID, varLonID, "units", "degree_east")

status = nf90_def_var(ncID,'latitude',NF90_FLOAT,(/ LatDimID /),varLatID)
status = nf90_put_att(ncID, varLatID, "standard_name", "latitude")
status = nf90_put_att(ncID, varLatID, "units", "degree_north")

status = nf90_def_var(ncID,'time',NF90_FLOAT, (/ TimeDimID /), varTimeID)
status = nf90_put_att(ncID, varTimeID, "axis", "T")
status = nf90_put_att(ncID, varTimeID, "calendar", "standard")
status = nf90_put_att(ncID, varTimeID, "long_name", "time")
status = nf90_put_att(ncID, varTimeID, "units", "hours since &
                                            1979-01-01 00:00:00 UTC")

! Global Attributes -----------------------------------------------
status = nf90_put_att(ncID, NF90_GLOBAL, 'Conventions', 'CF-1.0')
status = nf90_put_att(ncID, NF90_GLOBAL, 'title', cfouttitle)
status = nf90_put_att(ncID, NF90_GLOBAL, 'source', cfoutsource)
status = nf90_put_att(ncID, NF90_GLOBAL, 'institution', cfoutinstit)
status = nf90_put_att(ncID, NF90_GLOBAL, 'comment', cfoutcomment)


! Specific variable -----------------------------------------------
status = nf90_def_var(ncID,trim(outvarname),NF90_FLOAT,&
                   (/ LonDimID, LatDimID, varTimeID /),varPVID)
!status = nf90_def_var(ncID,trim(outvarname),NF90_FLOAT,&
!                   (/ LatDimID, varTimeID /),varPVID)
status = nf90_put_att(ncID, varPVID, "standard_name",trim(outstandvarname))
status = nf90_put_att(ncID, varPVID, "units", trim(outvarunit)) 
status = nf90_put_att(ncID, varPVID, '_FillValue', -999.99) 

! END variable definitions.

status = nf90_enddef(ncID)
IF (status.GT.0) THEN
   print*, 'An error occurred while attempting to ', &
        'finish definition mode.'
   !GOTO 920
ENDIF

! write DATA
! -------------------------------------------------------------------
status = nf90_put_var(ncID,varLonID,longitude)
status = nf90_put_var(ncID,varLatID,latitude)
!status = nf90_put_var(ncID,varTimeID, times)
status = nf90_put_var(ncID,varTimeID, &
                      times((persistence+1):(nTimes-persistence)))


!  Write block
!status = nf90_put_var(ncID,varPVID, arr(:,:,:))
status = nf90_put_var(ncID,varPVID, &
                      arr(:,:,(persistence+1):(nTimes-persistence)))

status = nf90_close(ncID)
IF (status.NE.0) THEN
   WRITE(0,*) trim(nf90_strerror(status))
   WRITE(0,*) 'An error occurred while attempting to ', &
              'close the netcdf file.'
   WRITE(0,*) 'in clscdf_CF'
ENDIF

close(ncID)


END


