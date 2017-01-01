#!/usr/bin/perl -w
#

=head1 $0

=head1 SYNOPSIS

 00-baseline.pl dataset [options] 

 Options:
  --help
  --sim=cos|dot    similarity used: dot product, cosine (default).

=cut

use Getopt::Long qw(:config auto_help); 
use Pod::Usage; 
use lib "." ;
use Similarity ;
use strict ;


# default parameter 
my $SIM = 'cos';

GetOptions("sim=s" => \$SIM,
	   )
    or
    pod2usage() ;

$Similarity::SIM  = $SIM ;               # Similarity package parameter
pod2usage() if not availablesim($SIM) ;  # check if $SIM is available

$Similarity::DICT = "dummyNotUsed" ;


# read input pairs
open(I,$ARGV[0]) or die $! ;
while (<I>) {
    chop ; 
    #next if /^#/ ;
    my($s1,$s2) = split(/\t/,$_) ;
    die "I need sentence pairs in input\n" if ! $s1 or ! $s2 ;
    my $vectors1 = makevector($s1) ;
    my $vectors2 = makevector($s2) ;
    printf "%.10f\n", similarity($vectors1,$vectors2) ;       
}



sub makevector {
    my ($s) = @_ ;
    my $vector = {} ;
    foreach my $token (split(/\s+/,$s)) {
	$vector->{$token} = 1 ;
    }
    return $vector ;
}
