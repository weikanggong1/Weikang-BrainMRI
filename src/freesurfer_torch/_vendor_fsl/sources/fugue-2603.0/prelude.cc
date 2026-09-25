/*  prelude.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000/2001 University of Oxford  */

/*  CCOPYRIGHT  */


#include "utils/options.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

#include "unwarpfns.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace FUGUE;


string title="prelude (Version 2.0 in c# minor)\nPhase Region Expanding Labeller for Unwrapping Discrete Estimates\nCopyright(c) 2001, University of Oxford (Mark Jenkinson)";
string examples="prelude -c <rawphase> -o <unwrappedphase> [options]\nprelude -p <phasevol> -a <absvol> -o <unwrappedphase> [options]";

Option<bool> verbose(string("-v,--verbose"), false,
		     string("switch on diagnostic messages"),
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<bool> labelslices(string("--labelslices"), false,
		    string("does label processing in 2D (slice at a time)"),
		    false, no_argument);
Option<bool> allslices(string("-s,--slices"), false,
		       string("does all processing in 2D (slice at a time)"),
		       false, no_argument);
Option<bool> force3D(string("-f,--force3D"), false,
		     string("forces all processing to be full 3D"),
		     false, no_argument);
Option<bool> removeramps(string("--removeramps"), false,
		     string("remove phase ramps during unwrapping"),
		     false, no_argument);
Option<int> num_phase_split(string("-n,--numphasesplit"), 8,
			      string("number of phase partitions to use"),
			      false, requires_argument);
Option<int> startimno(string("--start"), 0,
		      string("first image number to process (default 0)"),
		      false, requires_argument);
Option<int> endimno(string("--end"), 0,
		    string("final image number to process (default Inf)"),
		    false, requires_argument);
Option<float> thresh(string("-t,--thresh"), 0,
		    string("intensity threshold for masking"),
		    false, requires_argument);
Option<string> complexvol(string("-c,--complex"), string(""),
			  string("filename of complex phase input volume"),
			  false, requires_argument);
Option<string> absvol(string("-a,--abs"), string(""),
		      string("filename of absolute input volume"),
		      false, requires_argument);
Option<string> phasevol(string("-p,--phase"), string(""),
			string("filename of raw phase input volume"),
			false, requires_argument);
Option<string> uphasevol(string("-o,--out,-u,--unwrap"), string(""),
			 string("filename for saving the unwrapped phase output"),
			 true, requires_argument);
Option<string> rawphasevol(string("-r,--rawphase"), string(""),
			   string("filename for saving the raw phase output"),
			   false, requires_argument);
Option<string> labelvol(string("-l,--labels"), string(""),
			   string("filename for saving the area labels output"),
			   false, requires_argument);
Option<string> maskvol(string("--savemask"), string(""),
			   string("filename for saving the mask volume"),
			   false, requires_argument);
Option<string> inmask(string("-m,--mask"), string(""),
		      string("filename of mask input volume"),
		      false, requires_argument);

////////////////////////////////////////////////////////////////////////////


void wrap(volume<float>& phasevol)
{
  for (int z=phasevol.minz(); z<=phasevol.maxz(); z++) {
    for (int y=phasevol.miny(); y<=phasevol.maxy(); y++) {
      for (int x=phasevol.minx(); x<=phasevol.maxx(); x++) {
	phasevol(x,y,z) = wrap(phasevol(x,y,z));
      }
    }
  }
}


int do_unwrapping()
{

  volume4D<float> phasemaps, absmaps, masks;

  if (complexvol.set()) {
    volume4D<float> rvol, ivol;
    read_complexvolume(rvol,ivol,complexvol.value());
    for (int n=0; n<rvol.tsize(); n++) {
      phasemaps.addvolume(phase(rvol[n],ivol[n]));
      absmaps.addvolume(abs(rvol[n],ivol[n]));
    }
    phasemaps.setTR(rvol.TR());
  } else {
    if (verbose.value()) cout << "Loading volumes" << endl;
    read_volume4D(phasemaps,phasevol.value());
    if (verbose.value()) cout << "Phase loaded" << endl;
    read_volume4D(absmaps,absvol.value());
    if (verbose.value()) cout << "Magnitude loaded" << endl;
  }

  bool threshset = thresh.set();
  float threshval = thresh.value();
  int imgstart=0, imgend, maskend=0;
  imgend = phasemaps.maxt();
  if (startimno.set())  imgstart = startimno.value();
  if (endimno.set())    imgend = endimno.value();
  if (inmask.set()) {
    read_volume4D(masks,inmask.value());
    if (verbose.value()) cout << "Mask loaded" << endl;
    maskend = masks.maxt();
    if (!threshset) {
      threshset = true;
      threshval = 0.5;
    }
  }

  // sanity check on phase values
  for (int n=imgstart; n<=imgend; n++) {
    if ((phasemaps[n].max() - phasemaps[n].min())>3.0*M_PI) {
      cerr << "ERROR: input phase image exceeds allowable phase range."
	   << endl << "Allowable range is 6.283 radians.  Image range is: "
	   << phasemaps[n].max() - phasemaps[n].min() << " radians."
	   << endl << "Aborting." << endl;
      return -1;
    }
    if ( (phasemaps[n].max()>1.001*M_PI) || (phasemaps[n].min()<-1.001*M_PI) )
      {
	if (verbose.value())
	  { cout << "Rewrapping phase range to [-pi,pi]" << endl; }
	ShadowVolume<float> tmpVol(phasemaps[n]);
	  wrap(tmpVol);
	//wrap(phasemaps[n]);
      }
  }


  bool label_2D = labelslices.value();
  bool unwrap_2D = allslices.value();
  // The automatic behaviour is to use 2D labels for large images
  if ((phasemaps[imgstart].xsize() > 64) || (phasemaps[imgstart].ysize() > 64))
    {
      label_2D = true;
    }
  if (force3D.value()) { label_2D = false;  unwrap_2D = false; }

  volume4D<float> uphase;

  for (int n=imgstart; n<=imgend; n++) {

    // Make mask
    volume<float> mask;
    if (inmask.set()) {
      mask = masks[Min(n,maskend)];
    } else {
      mask = absmaps[n];
    }

    if (!threshset) { threshval = basic_mask_threshold(mask); }

    if (unwrap_2D) {
      mask = make_head_mask2D(mask,threshval);
    } else {
      mask = make_head_mask(mask,threshval);
    }
    if (maskvol.set()) { save_volume(mask,maskvol.value()); }

    // Remove phase ramps
    ColumnVector ramps;
    if (removeramps.value()) {
      if (verbose.value()) { cout << "Removing linear ramps" << endl; }
      ramps = estimate_linear_ramps(phasemaps[n],mask);
      if (verbose.value()) { cout << "Estimated linear ramps" << endl; }
      ShadowVolume<float> tmpVol(phasemaps[n]);
      remove_linear_ramps(ramps,tmpVol,mask);
      if (verbose.value()) { cout << "Removed linear ramps" << endl; }
      if (verbose.value()) { cout << "Ramp values: " << ramps << endl; }
      if (verbose.value()) { save_volume(phasemaps[n],"grot"); }
    }

    // Make labels
    volume<int> label;
    if (verbose.value())
      {cout << "Number of phase splits = " << num_phase_split.value() << endl;}
    if (unwrap_2D) {
      label = find_phase_labels2D(phasemaps[n],mask,
				  num_phase_split.value(),false);
    } else if (label_2D) {
      label = find_phase_labels2D(phasemaps[n],mask,
				  num_phase_split.value(),true);
    } else {
      label = find_phase_labels(phasemaps[n],mask,num_phase_split.value());
    }
    if (labelvol.set()) { save_volume(label,labelvol.value()); }

    // Unwrap phase
    volume<float> uph;
    if (unwrap_2D) {
      uph = unwrap2D(phasemaps[n],label,verbose.value());
    } else {
      uph = unwrap(phasemaps[n],label,verbose.value());
    }

    // Restore phase ramps
    if (removeramps.value()) {
      restore_linear_ramps(ramps,uph,mask);
      ShadowVolume<float> tmpVol(phasemaps[n]);
      restore_linear_ramps(ramps,tmpVol,mask);
      if (verbose.value()) { cout << "Removed linear ramps" << endl; }
    }

    uphase.addvolume(uph);
  }

  // Save outputs
  uphase.setTR(phasemaps.TR());
  save_volume4D(uphase,uphasevol.value());

  if (rawphasevol.set()) {
    save_volume4D(phasemaps,rawphasevol.value());
  }

  return 0;
}




int main(int argc,char *argv[])
{

  Tracer tr("main");
  OptionParser options(title, examples);

  try {
    options.add(complexvol);
    options.add(absvol);
    options.add(phasevol);
    options.add(uphasevol);
    options.add(num_phase_split);
    options.add(labelslices);
    options.add(allslices);
    options.add(force3D);
    options.add(thresh);
    options.add(inmask);
    options.add(startimno);
    options.add(endimno);
    options.add(maskvol);
    options.add(rawphasevol);
    options.add(labelvol);
    options.add(removeramps);
    options.add(verbose);
    options.add(help);

    options.parse_command_line(argc, argv);

    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }

    if ( ( (complexvol.set()) && (absvol.set()) ) ||
	 ( (complexvol.set()) && (phasevol.set()) ) )
      {
	options.usage();
	cerr << endl
	     << "Cannot specify both --complex AND --phase or --abs."
	     << endl;
	exit(EXIT_FAILURE);
      }

    if ( ( (phasevol.set()) && (absvol.unset()) ) ||
	 ( (phasevol.unset()) && (absvol.set()) ) )
      {
	options.usage();
	cerr << endl
	     << "Both --phase AND --abs must be used together."
	     << endl;
	exit(EXIT_FAILURE);
      }

    if (num_phase_split.value() < 2)
      {
	options.usage();
	cerr << endl
	     << "Always set --numphasesplit greater than 1."
	     << endl << "NOT " << num_phase_split.value() << endl;
	exit(EXIT_FAILURE);
      }

  }  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }

  return do_unwrapping();
}
