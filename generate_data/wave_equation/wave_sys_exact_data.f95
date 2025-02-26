program wave_sys_singleP_exact
implicit none
integer :: nx, i
real, allocatable, dimension(:) :: x,p1, f,g,df
real :: pi, leftBoundary, rightBoundary, h
real, dimension(5) :: a

pi = 4*atan(1.0)

leftBoundary  = -pi
rightBoundary = pi

nx = 200
allocate(x(0:nx+1))
allocate(f(0:nx+1))
allocate(g(0:nx+1))
allocate(df(0:nx+1))
h = (rightBoundary-leftBoundary)/nx

do i=0,nx+1
   x(i) = leftBoundary-h/2+h*i
end do

call exact_wave_sys_solution(x,nx,h)

end program wave_sys_singleP_exact


subroutine exact_wave_sys_solution(x, nx, h)
implicit none
integer :: nx,k,l,i,nt,n_ex
real, dimension(0:nx+1) :: x, u
real :: dt, t_max, time, s, h, c, t, start, now, sec_left
real, allocatable, dimension(:) :: a1,b1,a2,b2
   !open(0,file='wave_ux_Exact_Solution_reference.dat')
   !open(1,file='wave_ut_Exact_solution_reference.dat')
   open(0,file='wave_ux0.dat')
   open(1,file='wave_ut0.dat')
   open(10,file='wave_ux1.dat')
   open(11,file='wave_ut1.dat')
   call init_random_seed()
   
   n_ex= 1000
   c  = 1
   dt = h/(2*c)
   k  = 3
   call cpu_time(start)
   nt = 1000
   do l=1,n_ex
      !call randomInt(1,3,k)
      !print*, k
      allocate(a1(k))
      allocate(b1(k))
      allocate(a2(k))
      allocate(b2(k))
      call waveSetUp_sys(a1,a2,b1,b2,k)
      call waveSolution_sys(x,nx,c,a1,b1,a2,b2,nt,dt,k,t)
      deallocate(a1)
      deallocate(b1)
      deallocate(a2)
      deallocate(b2)
      if (mod(l,10)==0) then
        call cpu_time(now)
        sec_left = ((now-start)/l)*(n_ex-l)
        print*, "time left is ",sec_left, "seconds"
        print*, l, '/' , n_ex
      endif
   end do 
   print*, "total time = ",now-start
   close(0)
   close(1)
   close(10)
   close(11)
end subroutine


subroutine waveSetUp_sys(a1,a2,b1,b2,k)
implicit none
integer :: k
real, dimension(k) :: a1, b1, a2, b2
real :: t, c, dt
   call random_number(a1)
   call random_number(b1)
   call random_number(a2)
   call random_number(b2)
   a1 = 2*a1-1
   b1 = 2*b1-1
   a2 = 2*a2-1
   b2 = 2*b2-1
   
   !a1 = [.6 , .8 , -.7]
   !b1 = [-.4, 1.0, -0.4]
   !a2 = [.1 , .1 , .7]
   !b2 = [.4 , -.7, -.4]
end subroutine


subroutine waveSolution_sys(x,nx,c,a1,b1,a2,b2,nt,dt,k,t) 
implicit none
integer :: nx, nt, i, k 
real :: speed, dt, t, c
real, dimension(0:nx+1) :: x, v, w, v0, w0, f, df_p, df_m, g_p, g_m
real, dimension(k) :: a1, b1, a2, b2
   t = 0.0
   call dphi1(x,nx,a1,b1,df_m,k)
   call dphi1(x,nx,a1,b1,df_p,k)
   call phi2(x,nx,a2,b2,g_m,k)
   call phi2(x,nx,a2,b2,g_p,k)

   v0 = .5*(df_m+df_p)+(1.0/(2.0*c))*(g_p-g_m)
   w0 = (c/2.0)*(df_p-df_m)+.5*(g_p+g_m)
   !write(0,*) v0(1:n)
   !write(1,*) w0(1:n)
   do i = 1,nt
      t = t+dt
      call dphi1(x-c*t,nx,a1,b1,df_m,k)
      call dphi1(x+c*t,nx,a1,b1,df_p,k)
      call phi2(x-c*t,nx,a2,b2,g_m,k)
      call phi2(x+c*t,nx,a2,b2,g_p,k)
      
      v = .5*(df_m+df_p)+(1.0/(2.0*c))*(g_p-g_m)
      w = (c/2.0)*(df_p-df_m)+.5*(g_p+g_m)
      
      call writeSolution(v0(1:nx),w0(1:nx),v(1:nx),w(1:nx),nx)
      v0 = v
      w0 = w
      !write(0,*) v(1:n)
      !write(1,*) w(1:n)
   end do
end subroutine


subroutine phi1(x,nx,a,b,f,k)
implicit none
integer :: nx, k, i
real, dimension(k) :: a, b
real, dimension(0:nx+1) :: x, f 
   f(:) = 0
   do i = 1,k
      f = f+(1.0/i)*(a(i)*cos(i*x)+b(i)*sin(i*x))
   end do
end subroutine


subroutine dphi1(x,nx,a,b,df,k)
implicit none
integer :: nx, k, i
real, dimension(k) :: a, b
real, dimension(0:nx+1) :: x, df 
   df(:) = 0
   do i=1,k
      df = df+(-a(i)*sin(i*x)+b(i)*cos(i*x))
   end do
end subroutine


subroutine phi2(x,nx,a,b,g,k)
implicit none
integer :: nx, k, i
real, dimension(k) :: a, b
real, dimension(0:nx+1) :: x, g 
   g(:) = 0
   do i=1,k
      g = g+(1.0/i)*(a(i)*cos(i*x)+b(i)*sin(i*x))
   end do
end subroutine


subroutine randomInt(a,b,ri)
implicit none
integer :: a, b, ri
real :: r
   call init_random_seed()
   call random_number(r)
   ri = a+FLOOR((b-a+1)*r)
end subroutine


subroutine writeSolution(v0,w0,v1,w1,m)
implicit none
integer :: m,l,i,k
real, dimension(m) :: v0, w0, v1, w1
   k = 40
   l = 2
   do i = k,m-1,k
      write(0,*) v0(i-l:i+l)
      write(1,*) w0(i-l:i+l)
      write(10,*) v1(i)
      write(11,*) w1(i)
   end do
end subroutine


subroutine init_random_seed()
   use iso_fortran_env, only: int64
   implicit none
   integer, allocatable :: seed(:)
   integer :: i, n, un, istat, dt(8), pid
   integer(int64) :: t
          
   call random_seed(size = n)
   allocate(seed(n))
   ! First try if the OS provides a random number generator
   open(newunit=un, file="/dev/urandom", access="stream", &
      form="unformatted", action="read", status="old", iostat=istat)
   if (istat == 0) then
      read(un) seed
      close(un)
   else
      ! Fallback to XOR:ing the current time and pid. The PID is
      ! useful in case one launches multiple instances of the same
      ! program in parallel.
      call system_clock(t)
      if (t == 0) then
         call date_and_time(values=dt)
         t = (dt(1) - 1970) * 365_int64 * 24 * 60 * 60 * 1000 &
             + dt(2) * 31_int64 * 24 * 60 * 60 * 1000 &
             + dt(3) * 24_int64 * 60 * 60 * 1000 &
             + dt(5) * 60 * 60 * 1000 &
             + dt(6) * 60 * 1000 + dt(7) * 1000 &
             + dt(8)
      end if
      pid = getpid()
      t = ieor(t, int(pid, kind(t)))
      do i = 1, n
         seed(i) = lcg(t)
      end do
   end if
   call random_seed(put=seed)
 contains
   ! This simple PRNG might not be good enough for real work, but is
   ! sufficient for seeding a better PRNG.
   function lcg(s)
       integer :: lcg
       integer(int64) :: s
       if (s == 0) then
           s = 104729
       else
           s = mod(s, 4294967296_int64)
       end if
           s = mod(s * 279470273_int64, 4294967291_int64)
           lcg = int(mod(s, int(huge(0), int64)), kind(0))
   end function lcg
end subroutine init_random_seed
